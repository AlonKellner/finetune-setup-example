"""Custom `transformers` trainer."""

import os
from typing import Any

import comet_ml
import torch
import wandb
from datasets import Dataset as HFDataset
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import has_length

from .length_sampler import CustomLengthGroupedSampler
from .training_args import CustomTrainingArguments


class CustomTrainer(Trainer):
    """A custom version of the trainer to make sure length sampling is mixed."""

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: CustomTrainingArguments | None = None,
        *arguments: Any,
        test_indices_order: list[int] | None = None,
        train_indices_order: list[int] | None = None,
        train_grouped_indices: list[list[int]] | None = None,
        test_grouped_indices: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, args, *arguments, **kwargs)  # type: ignore
        assert args is not None
        self.args = args

        self.test_indices_order = test_indices_order
        self.train_indices_order = train_indices_order

        self.test_grouped_indices = test_grouped_indices
        self.train_grouped_indices = train_grouped_indices

    def _get_train_sampler(self) -> torch.utils.data.Sampler | None:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if self.args.has_length_column:
                if hasattr(self.train_dataset, "metadata"):
                    lengths = [
                        v[self.args.length_column_name]
                        for k, v in self.train_dataset.metadata.items()  # type: ignore
                    ]
                else:
                    lengths = (
                        self.train_dataset[self.args.length_column_name]  # type: ignore
                        if self.args.length_column_name
                        in self.train_dataset.column_names  # type: ignore
                        else None
                    )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0]  # type: ignore
                if self.processing_class is not None
                else None
            )
            return CustomLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,  # type: ignore
                lengths=lengths,
                model_input_name=model_input_name,
                mega_batch_mult=self.args.mega_batch_mult,
                indices_order=self.train_indices_order,
                grouped_indices=self.train_grouped_indices,
            )

        else:
            return RandomSampler(self.train_dataset)  # type: ignore

    def _get_eval_sampler(  # type: ignore
        self, eval_dataset: HFDataset
    ) -> torch.utils.data.Sampler | None:
        if eval_dataset is None or not has_length(eval_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if self.args.has_length_column:
                lengths = (
                    eval_dataset[self.args.length_column_name]
                    if self.args.length_column_name in eval_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.tokenizer.model_input_names[0]
                if self.tokenizer is not None
                else None
            )
            return CustomLengthGroupedSampler(
                self.args.eval_batch_size,
                dataset=eval_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                indices_order=self.test_indices_order,
                grouped_indices=self.test_grouped_indices,
            )

        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return None


def train(trainer: Trainer) -> None:
    """Train with cometml and wandb."""
    comet_ml.login(project_name=os.getenv("WANDB_PROJECT"))
    wandb.init(dir="./.wandb")

    trainer.train()
    trainer.evaluate()

    comet_ml.end()
    wandb.finish()
