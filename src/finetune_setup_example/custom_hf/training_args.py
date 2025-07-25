"""Custom `transformers` training args."""

import os
from dataclasses import dataclass, field
from typing import Literal

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

    pretrained_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The max learning rate, for pretrained layers."},
    )
    pretrained_min_lr_ratio: float = field(
        default=0.0,
        metadata={"help": "The min learning rate ratio, for pretrained layers."},
    )
    pretrained_wait_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Wait period over wait_ratio fraction of total steps, for pretrained layers."
        },
    )
    pretrained_warmup_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Linear warmup over warmup_ratio fraction of total steps, for pretrained layers."
        },
    )
    pretrained_stable_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Stable period over stable_ratio fraction of total steps, for pretrained layers."
        },
    )
    pretrained_decay_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Cosine decay over decay_ratio fraction of total steps, for pretrained layers."
        },
    )

    adapter_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The max learning rate, for adapter layers."},
    )
    adapter_min_lr_ratio: float = field(
        default=0.0,
        metadata={"help": "The min learning rate ratio, for adapter layers."},
    )
    adapter_wait_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Wait period over wait_ratio fraction of total steps, for adapter layers."
        },
    )
    adapter_warmup_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Linear warmup over warmup_ratio fraction of total steps, for adapter layers."
        },
    )
    adapter_stable_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Stable period over stable_ratio fraction of total steps, for adapter layers."
        },
    )
    adapter_decay_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Cosine decay over decay_ratio fraction of total steps, for adapter layers."
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
    pretrained_learning_rate: float,
    pretrained_min_lr_ratio: float,
    pretrained_wait_ratio: float,
    pretrained_warmup_ratio: float,
    pretrained_stable_ratio: float,
    pretrained_decay_ratio: float,
    adapter_learning_rate: float,
    adapter_min_lr_ratio: float,
    adapter_wait_ratio: float,
    adapter_warmup_ratio: float,
    adapter_stable_ratio: float,
    adapter_decay_ratio: float,
    mega_batch_mult: int,
    dataloader_num_workers: int,
    fp16: bool,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    weight_decay: float,
    torch_compile: bool,
    sample_rate: int,
    eval_on_start: bool,
    architecture: Literal["wav2vec2", "w2v-bert2"],
    num_train_epochs: int | float | None = 3.0,
    num_training_steps: int | None = None,
    hp_set: dict | None = None,
) -> CustomTrainingArguments:
    """Create training arguments."""
    run_name = (
        target_hf_repo if (job_id := os.getenv("JOB_FULL_ID")) is None else job_id
    )

    global_batch_size = per_device_train_batch_size * num_devices
    accumulation_steps = effective_batch_size // global_batch_size

    if architecture == "wav2vec2":
        lengths_per_second = sample_rate
    elif architecture == "w2v-bert2":
        lengths_per_second = 50
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    per_device_train_batch_total_length = int(
        per_device_train_batch_total_seconds * lengths_per_second
    )
    per_device_eval_batch_total_length = int(
        per_device_eval_batch_total_seconds * lengths_per_second
    )

    if (num_training_steps is None) and (num_train_epochs is not None):
        num_training_steps = -1
    elif (num_train_epochs is None) and (num_training_steps is not None):
        num_train_epochs = 3.0
    else:
        raise ValueError(
            "Either `num_training_steps` or `num_train_epochs` must be provided, "
            "but not both."
        )

    training_args = CustomTrainingArguments(
        seed=seed,
        data_seed=data_seed,
        report_to=["comet_ml", "wandb"],
        output_dir=f".output/{target_hf_repo}",
        resume_from_checkpoint="last-checkpoint",
        hub_strategy="checkpoint",
        save_total_limit=2,
        push_to_hub=True,
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
        pretrained_learning_rate=pretrained_learning_rate,
        pretrained_min_lr_ratio=pretrained_min_lr_ratio,
        pretrained_wait_ratio=pretrained_wait_ratio,
        pretrained_warmup_ratio=pretrained_warmup_ratio,
        pretrained_stable_ratio=pretrained_stable_ratio,
        pretrained_decay_ratio=pretrained_decay_ratio,
        adapter_learning_rate=adapter_learning_rate,
        adapter_min_lr_ratio=adapter_min_lr_ratio,
        adapter_wait_ratio=adapter_wait_ratio,
        adapter_warmup_ratio=adapter_warmup_ratio,
        adapter_stable_ratio=adapter_stable_ratio,
        adapter_decay_ratio=adapter_decay_ratio,
        weight_decay=weight_decay,
        logging_nan_inf_filter=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        log_level="info",
        torch_compile=torch_compile,
        hp_set=hp_set,
    )

    return training_args
