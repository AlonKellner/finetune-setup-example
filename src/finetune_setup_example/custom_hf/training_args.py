"""Custom `transformers` training args."""

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


def create_training_arguments(
    seed: int,
    data_seed: int,
    target_hf_repo: str,
    num_train_epochs: int,
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
) -> CustomTrainingArguments:
    """Create training arguments."""
    global_batch_size = per_device_train_batch_size * num_devices
    accumulation_steps = effective_batch_size // global_batch_size

    per_device_train_batch_total_length = int(
        per_device_train_batch_total_seconds * sample_rate
    )
    per_device_eval_batch_total_length = int(
        per_device_eval_batch_total_seconds * sample_rate
    )

    num_training_steps = (
        train_size // effective_batch_size  # type: ignore
    ) * num_train_epochs

    lr_scheduler_kwargs = dict(num_decay_steps=int(decay_ratio * num_training_steps))

    training_args = CustomTrainingArguments(
        seed=seed,
        data_seed=data_seed,
        report_to=["comet_ml", "wandb"],
        output_dir=f".output/{target_hf_repo}",
        run_name=target_hf_repo,
        group_by_length=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        per_device_train_batch_total_length=per_device_train_batch_total_length,
        per_device_eval_batch_total_length=per_device_eval_batch_total_length,
        gradient_accumulation_steps=accumulation_steps,
        eval_strategy="steps",
        num_train_epochs=num_train_epochs,
        mega_batch_mult=mega_batch_mult,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        fp16=fp16,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        eval_on_start=True,
        logging_first_step=True,
        learning_rate=learning_rate,
        lr_scheduler_type="warmup_stable_decay",
        warmup_steps=int(warmup_ratio * num_training_steps),
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
    )

    return training_args
