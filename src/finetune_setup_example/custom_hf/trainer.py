"""Custom `transformers` trainer."""

import math
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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
from transformers.trainer_utils import SaveStrategy, has_length, seed_worker
from transformers.utils import is_datasets_available

from .length_sampler import CustomLengthGroupedSampler
from .training_args import CustomTrainingArguments


class WarmupStableDecayScheduler:
    """A custom learning rate scheduler that implements a warmup, stable, and decay phase."""

    def __init__(
        self,
        max_steps: int,
        min_lr_ratio: float,
        wait_ratio: float,
        warmup_ratio: float,
        stable_ratio: float,
        decay_ratio: float,
    ) -> None:
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.wait_steps = int(max_steps * wait_ratio)
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.stable_steps = int(max_steps * stable_ratio)
        self.decay_steps = int(max_steps * decay_ratio)

    def __call__(self, current_step: int) -> float:
        """Calculate the learning rate factor based on the current step."""
        if current_step < self.wait_steps:
            value = 0.0
        elif current_step < self.wait_steps + self.warmup_steps:
            progress = float(current_step - self.wait_steps) / float(
                max(1, self.warmup_steps)
            )
            value = max(0.0, progress)
        elif current_step < self.wait_steps + self.warmup_steps + self.stable_steps:
            value = 1.0
        elif (
            current_step
            < self.wait_steps + self.warmup_steps + self.stable_steps + self.decay_steps
        ):
            progress = float(
                current_step - self.wait_steps - self.warmup_steps - self.stable_steps
            ) / float(max(1, self.decay_steps))
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            value = max(0.0, factor)
        else:
            value = 0.0

        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * value


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

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """
        Setups the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Setups the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            opt_model = self.model
            decay_parameters = self.get_decay_parameter_names(opt_model)
            adapter_parameters = opt_model._get_adapters()
            grad_params = [
                (n, p) for n, p in opt_model.named_parameters() if p.requires_grad
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in grad_params
                        if (
                            ((n in decay_parameters) == is_decay)
                            and ((n in adapter_parameters) == is_adapter)
                        )
                    ],
                    "weight_decay": self.args.weight_decay if is_decay else 0.0,
                    "lr": (
                        self.args.adapter_learning_rate
                        if is_adapter
                        else self.args.pretrained_learning_rate
                    ),
                }
                for is_adapter in [True, False]
                for is_decay in [True, False]
            ]

            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Setups the scheduler.

        The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            adapter_scheduler = WarmupStableDecayScheduler(
                num_training_steps,
                self.args.adapter_min_lr_ratio,
                self.args.adapter_wait_ratio,
                self.args.adapter_warmup_ratio,
                self.args.adapter_stable_ratio,
                self.args.adapter_decay_ratio,
            )
            pretrained_scheduler = WarmupStableDecayScheduler(
                num_training_steps,
                self.args.pretrained_min_lr_ratio,
                self.args.pretrained_wait_ratio,
                self.args.pretrained_warmup_ratio,
                self.args.pretrained_stable_ratio,
                self.args.pretrained_decay_ratio,
            )
            if optimizer is None:
                optimizer = self.create_optimizer()
            self.lr_scheduler = LambdaLR(
                optimizer,
                [
                    adapter_scheduler if is_adapter else pretrained_scheduler
                    for is_adapter in [True, False]
                    for is_decay in [True, False]
                ],
                -1,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

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
            self.train_sampler = CustomLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,  # type: ignore
                lengths=lengths,
                model_input_name=model_input_name,
                mega_batch_mult=self.args.mega_batch_mult,
                indices_order=self.train_indices_order,
                grouped_indices=self.train_grouped_indices,
                batch_total_length=self.args.per_device_train_batch_total_length,
            )
            return self.train_sampler

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
            self.eval_sampler = CustomLengthGroupedSampler(
                self.args.eval_batch_size,
                dataset=eval_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                indices_order=self.eval_indices_order,
                grouped_indices=self.eval_grouped_indices,
                batch_total_length=self.args.per_device_eval_batch_total_length,
            )
            return self.eval_sampler

        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return None

    def _maybe_log_save_evaluate(
        self,
        tr_loss: float | torch.Tensor,
        grad_norm: float | torch.Tensor | None,
        model: nn.Module,
        trial: Any | None,
        epoch: float,  # noqa: ARG002
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,  # noqa: ARG002
    ) -> None:
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            all_learning_rates = [
                (last_lr.item() if torch.is_tensor(last_lr) else last_lr)
                for last_lr in self.lr_scheduler.get_last_lr()
            ]
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            for i, lr in enumerate(all_learning_rates):
                logs[f"learning_rate/{i}"] = lr

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )


def train(trainer: Trainer) -> None:
    """Train with cometml and wandb."""
    comet_ml.login(project_name=os.getenv("WANDB_PROJECT"))
    wandb.init(dir="./.wandb")

    trainer.train()  # type: ignore
    trainer.evaluate()  # type: ignore

    comet_ml.end()
    wandb.finish()
