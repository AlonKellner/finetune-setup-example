"""Custom `transformers` training args."""

import os
from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Custom training args."""

    mega_batch_mult: int = field(
        default=50, metadata={"help": "The mega batch multiple."}
    )

    has_length_column: bool = field(
        default=True,
        metadata={
            "help": "Sets whether the dataset has a length column for length sampling."
        },
    )

    per_device_train_batch_total_length: int | None = field(
        default=None,
        metadata={
            "help": "The total length of the batch for length sampling during training. "
            "If None, length sampling is disabled."
        },
    )

    per_device_eval_batch_total_length: int | None = field(
        default=None,
        metadata={
            "help": "The total length of the batch for length sampling during evaluation. "
            "If None, length sampling is disabled."
        },
    )

    hp_set: dict | None = field(
        default=None,
        metadata={
            "help": ("Extra hyper-parameter set with all of the user defined values.")
        },
    )


def create_training_arguments(
    seed: int,
    data_seed: int,
    target_hf_repo: str,
    effective_batch_size: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    per_device_train_batch_total_seconds: float,
    per_device_eval_batch_total_seconds: float,
    num_devices: int,
    warmup_ratio: float,
    decay_ratio: float,
    learning_rate: float,
    mega_batch_mult: int,
    dataloader_num_workers: int,
    fp16: bool,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    weight_decay: float,
    torch_compile: bool,
    train_size: int,
    sample_rate: int,
    eval_on_start: bool,
    num_train_epochs: int | float | None = 3.0,
    num_training_steps: int | None = None,
    steps_per_epoch: int | None = None,
    hp_set: dict | None = None,
) -> CustomTrainingArguments:
    """Create training arguments."""
    run_name = (
        target_hf_repo if (job_id := os.getenv("JOB_FULL_ID")) is None else job_id
    )

    global_batch_size = per_device_train_batch_size * num_devices
    accumulation_steps = effective_batch_size // global_batch_size

    per_device_train_batch_total_length = int(
        per_device_train_batch_total_seconds * sample_rate
    )
    per_device_eval_batch_total_length = int(
        per_device_eval_batch_total_seconds * sample_rate
    )

    if steps_per_epoch is None:
        steps_per_epoch = train_size // effective_batch_size

    if (num_training_steps is None) and (num_train_epochs is not None):
        _num_training_steps = num_train_epochs * steps_per_epoch
        _num_train_epochs = num_train_epochs
        num_training_steps = -1
    elif (num_train_epochs is None) and (num_training_steps is not None):
        _num_train_epochs = num_training_steps // steps_per_epoch
        _num_training_steps = num_training_steps
        num_train_epochs = 3.0
    else:
        raise ValueError(
            "Either `num_training_steps` or `num_train_epochs` must be provided, "
            "but not both."
        )

    lr_scheduler_kwargs = dict(num_decay_steps=int(decay_ratio * _num_training_steps))
    warmup_steps = int(warmup_ratio * _num_training_steps)
    print(
        f"run name: {run_name}\n",
        f"num_training_steps: {num_training_steps}\n"
        f"_num_training_steps: {_num_training_steps}\n"
        f"num_train_epochs: {num_train_epochs}\n"
        f"_num_train_epochs: {_num_train_epochs}\n"
        f"steps_per_epoch: {steps_per_epoch}\n"
        f"warmup steps: {warmup_steps}\n"
        f"decay steps: {lr_scheduler_kwargs['num_decay_steps']}\n"
        f"learning rate: {learning_rate}\n"
        f"batch size: {global_batch_size}\n"
        f"accumulation steps: {accumulation_steps}\n"
        f"effective batch size: {effective_batch_size}\n"
        f"per device train batch size: {per_device_train_batch_size}\n"
        f"per device eval batch size: {per_device_eval_batch_size}\n"
        f"per device train batch total length: {per_device_train_batch_total_length}\n"
        f"per device eval batch total length: {per_device_eval_batch_total_length}\n"
        f"mega batch mult: {mega_batch_mult}\n",
    )

    training_args = CustomTrainingArguments(
        seed=seed,
        data_seed=data_seed,
        report_to=["comet_ml", "wandb"],
        output_dir=f".output/{target_hf_repo}",
        run_name=run_name,
        group_by_length=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        per_device_train_batch_total_length=per_device_train_batch_total_length,
        per_device_eval_batch_total_length=per_device_eval_batch_total_length,
        gradient_accumulation_steps=accumulation_steps,
        eval_strategy="steps",
        num_train_epochs=num_train_epochs,
        max_steps=num_training_steps,
        mega_batch_mult=mega_batch_mult,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        fp16=fp16,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        eval_on_start=eval_on_start,
        logging_first_step=True,
        learning_rate=learning_rate,
        lr_scheduler_type="warmup_stable_decay",
        warmup_steps=warmup_steps,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        weight_decay=weight_decay,
        save_total_limit=2,
        push_to_hub=False,
        logging_nan_inf_filter=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        log_level="info",
        torch_compile=torch_compile,
        hp_set=hp_set,
    )

    return training_args
