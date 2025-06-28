"""Custom `transformers` trainer."""

import os
from collections.abc import Callable
from functools import partial
from typing import Any

import comet_ml
import datasets
import torch
import wandb
from datasets import Dataset as HFDataset
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)
from transformers import Trainer
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import has_length, seed_worker
from transformers.utils import is_datasets_available

from .length_sampler import CustomLengthGroupedSampler
from .training_args import CustomTrainingArguments


class CustomTrainer(Trainer):  # type: ignore
    """A custom version of the trainer to make sure length sampling is mixed."""

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: CustomTrainingArguments | None = None,
        *arguments: Any,
        train_indices_order: list[int] | None = None,
        train_grouped_indices: list[list[int]] | None = None,
        eval_indices_order: list[int] | None = None,
        eval_grouped_indices: list[list[int]] | None = None,
        eval_data_collator: DataCollator | None = None,  # type: ignore
        eval_processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, args, *arguments, **kwargs)  # type: ignore
        assert args is not None
        self.args = args

        self.eval_indices_order = eval_indices_order
        self.train_indices_order = train_indices_order

        self.eval_grouped_indices = eval_grouped_indices
        self.train_grouped_indices = train_grouped_indices

        default_collator = (
            DataCollatorWithPadding(eval_processing_class)  # type: ignore
            if eval_processing_class is not None
            and isinstance(
                eval_processing_class,
                PreTrainedTokenizerBase | SequenceFeatureExtractor,
            )
            else default_data_collator
        )
        self.eval_data_collator = (
            eval_data_collator if eval_data_collator is not None else default_collator
        )
        self.eval_processing_class = eval_processing_class
        self.is_batch_sampler = True

    def get_train_dataloader(self) -> DataLoader:
        """
        Return the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_collated_dataloader(
            dataset=self.train_dataset,  # type: ignore
            data_collator=self.data_collator,
            description="Training",
            batch_size=self._train_batch_size,  # type: ignore
            sampler_fn=self._get_train_sampler,  # type: ignore
            is_training=True,
        )

    def get_eval_dataloader(
        self, eval_dataset: str | Dataset | None = None
    ) -> DataLoader:
        """
        Return the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]  # type: ignore
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        return self._get_collated_dataloader(
            dataset=eval_dataset,  # type: ignore
            data_collator=self.eval_data_collator,
            description="Evaluation",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,  # type: ignore
            dataloader_key=dataloader_key,
        )

    def _get_collated_dataloader(
        self,
        dataset: Dataset,
        data_collator: DataCollator,  # type: ignore
        description: str,
        batch_size: int,
        sampler_fn: Callable[[Dataset], torch.utils.data.Sampler] | None = None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)  # type: ignore
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description=description
            )

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if self.is_batch_sampler and (sampler_fn is not None):
                dataloader_params["batch_sampler"] = sampler_fn(dataset)
                del dataloader_params["batch_size"]
            else:
                if sampler_fn is not None:
                    dataloader_params["sampler"] = sampler_fn(dataset)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return self.accelerator.prepare(dataloader)

    def _get_train_sampler(
        self, train_dataset: Dataset | IterableDataset | HFDataset | None = None
    ) -> torch.utils.data.Sampler | None:
        if train_dataset is None:
            train_dataset = self.train_dataset

        if train_dataset is None or not has_length(train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if self.args.has_length_column:
                if hasattr(train_dataset, "metadata"):
                    lengths = [
                        v[self.args.length_column_name]
                        for k, v in train_dataset.metadata.items()  # type: ignore
                    ]
                else:
                    lengths = (
                        train_dataset[self.args.length_column_name]  # type: ignore
                        if self.args.length_column_name in train_dataset.column_names  # type: ignore
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
                dataset=train_dataset,  # type: ignore
                lengths=lengths,
                model_input_name=model_input_name,
                mega_batch_mult=self.args.mega_batch_mult,
                indices_order=self.train_indices_order,
                grouped_indices=self.train_grouped_indices,
                batch_total_length=self.args.per_device_train_batch_total_length,
            )

        else:
            return RandomSampler(train_dataset)  # type: ignore

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
                self.eval_processing_class.model_input_names[0]  # type: ignore
                if self.eval_processing_class is not None
                else None
            )
            return CustomLengthGroupedSampler(
                self.args.eval_batch_size,
                dataset=eval_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                indices_order=self.eval_indices_order,
                grouped_indices=self.eval_grouped_indices,
                batch_total_length=self.args.per_device_eval_batch_total_length,
            )

        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return None


def train(trainer: Trainer) -> None:
    """Train with cometml and wandb."""
    comet_ml.login(project_name=os.getenv("WANDB_PROJECT"))
    wandb.init(dir="./.wandb")

    trainer.train()  # type: ignore
    trainer.evaluate()  # type: ignore

    comet_ml.end()
    wandb.finish()
