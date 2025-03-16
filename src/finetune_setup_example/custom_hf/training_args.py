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
