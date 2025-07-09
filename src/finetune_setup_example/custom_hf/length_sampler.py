"""A custom version of the `transformers` length grouped sampler."""

from collections.abc import Iterator

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch import Generator
from torch.utils.data import Dataset, Sampler
from transformers import BatchEncoding
from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_length_grouped_batches(
    lengths: list[int],
    batch_size: int,
    mega_batch_mult: int | None = None,
    generator: Generator | None = None,
    indices_order: list[int] | None = None,
    grouped_indices: list[list[int]] | None = None,
    batch_total_length: int | None = None,
) -> list[list[int]]:
    """Get shuffled megabatches, A custom version. First is still largest.

    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    indices, grouped_indices = shuffle_indices_in_groups(
        lengths, generator, indices_order, grouped_indices
    )
    megabatches = generate_megabatches(indices, batch_size, mega_batch_mult)
    megabatches = move_longest_item_to_first_megabatch(megabatches, lengths)
    megabatches = sort_each_megabatch_by_length(megabatches, lengths)

    indices = [i for megabatch in megabatches for i in megabatch]

    batches = generate_total_length_batches(
        indices, lengths, batch_size, batch_total_length
    )
    batches = shuffle_batches_in_groups(batches, grouped_indices, generator)

    indices = [i for batch in batches for i in batch]

    if indices_order is not None:
        assert len(indices) == len(indices_order)
        assert len(set(indices)) == len(indices_order)
        assert max(indices) == len(indices_order) - 1
    else:
        assert len(indices) == len(lengths)
        assert len(set(indices)) == len(lengths)
        assert max(indices) == len(lengths) - 1
    assert min(indices) == 0
    return batches


def sort_each_megabatch_by_length(
    megabatches: list[list[int]], lengths: list[int]
) -> list[list[int]]:
    """Sort each megabatch by length."""
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=((mega_i % 2) == 0))
        for mega_i, megabatch in enumerate(megabatches)
    ]

    return megabatches


def generate_megabatches(
    indices: list[int], batch_size: int, mega_batch_mult: int
) -> list[list[int]]:
    """Generate megabatches of indices."""
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [
        indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)
    ]

    return megabatches


def move_longest_item_to_first_megabatch(
    megabatches: list[list[int]], lengths: list[int]
) -> list[list[int]]:
    """Move the longest item to the first megabatch."""
    megabatch_maximums = [
        max([lengths[i] for i in megabatch]) for megabatch in megabatches
    ]
    max_megabatch_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    assert isinstance(max_megabatch_idx, int)
    max_item_idx = torch.argmax(
        torch.tensor([lengths[i] for i in megabatches[max_megabatch_idx]])
    ).item()
    assert isinstance(max_item_idx, int)
    first_item_idx = torch.argmax(
        torch.tensor([lengths[i] for i in megabatches[0]])
    ).item()
    assert isinstance(first_item_idx, int)
    megabatches[0][first_item_idx], megabatches[max_megabatch_idx][max_item_idx] = (
        megabatches[max_megabatch_idx][max_item_idx],
        megabatches[0][first_item_idx],
    )
    return megabatches


def generate_total_length_batches(
    indices: list[int],
    lengths: list[int],
    batch_size: int,
    batch_total_length: int | None = None,
) -> list[list[int]]:
    """Generate batches of indices."""
    if batch_total_length is None:
        sizes = np.array([1 for _ in indices])
        batch_total_size = batch_size
    else:
        assert all(length <= batch_total_length for length in lengths), (
            "All lengths must be less than or equal to batch_total_length."
        )
        sizes = np.array([lengths[i] for i in indices])
        batch_total_size = batch_total_length

    i = 0
    prev_i = 0
    batches = []
    while i < len(indices):
        current_total_size = sizes[prev_i:i].max(initial=0) * (i - prev_i)
        if current_total_size > batch_total_size:
            batches.append(indices[prev_i : i - 1])
            prev_i = i - 1
        else:
            i += 1
    batches.append(indices[prev_i:])
    return batches


def shuffle_batches_in_groups(
    batches: list[list[int]],
    grouped_indices: list[list[int]],
    generator: Generator | None,
) -> list[list[int]]:
    """Shuffle batches."""
    first_batch = batches[0]
    batches = batches[1:]

    indices_to_group_idx = {
        i: group_i
        for group_i, group_idx in enumerate(grouped_indices)
        for i in group_idx
    }
    batches_to_group_idx = [
        tuple(set([indices_to_group_idx[i] for i in batch])) for batch in batches
    ]

    grouped_batch_indices = [[]]
    previous_group_idx = batches_to_group_idx[0]
    for batch_i, group_idx in enumerate(batches_to_group_idx):
        if group_idx != previous_group_idx:
            grouped_batch_indices.append([])
            previous_group_idx = group_idx
        grouped_batch_indices[-1].append(batch_i)

    shuffled_batch_indices = []
    for batch_indices in grouped_batch_indices:
        perm = torch.randperm(len(batch_indices), generator=generator).tolist()
        shuffled_indices = [batch_indices[i] for i in perm]
        shuffled_batch_indices.append(shuffled_indices)

    batches_perm = [
        batch_i for batch_idx in shuffled_batch_indices for batch_i in batch_idx
    ]

    batches = [batches[batch_i] for batch_i in batches_perm]
    batches = [first_batch, *batches]
    return batches


def shuffle_indices_in_groups(
    lengths: list[int],
    generator: Generator | None = None,
    indices_order: list[int] | None = None,
    grouped_indices: list[list[int]] | None = None,
) -> tuple[list[int], list[list[int]]]:
    """Shuffle indices and group them."""
    if grouped_indices is not None:
        indices = []
        if indices_order is not None:
            grouped_indices = [
                [item for item in group if item in indices_order]
                for group in grouped_indices
            ]
            grouped_indices = [group for group in grouped_indices if (len(group) > 0)]
        for group in grouped_indices:
            group_perm = torch.randperm(len(group), generator=generator)
            group_indices = [group[i] for i in group_perm]
            indices.extend(group_indices)
    elif indices_order is not None:
        perm = torch.randperm(len(indices_order), generator=generator).tolist()
        indices = [indices_order[i] for i in perm]
    else:
        indices = torch.randperm(len(lengths), generator=generator).tolist()

    if grouped_indices is None:
        grouped_indices = [indices]
    return indices, grouped_indices


class CustomLengthGroupedSampler(Sampler[list[int]]):
    """Custom sampler for random order."""

    def __init__(
        self,
        batch_size: int,
        dataset: Dataset | HFDataset | None = None,
        lengths: list[int] | None = None,
        model_input_name: str | None = None,
        generator: Generator | None = None,
        mega_batch_mult: int | None = None,
        indices_order: list[int] | None = None,
        grouped_indices: list[list[int]] | None = None,
        batch_total_length: int | None = None,
    ) -> None:
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = (
                model_input_name if model_input_name is not None else "input_ids"
            )
            assert isinstance(dataset, Dataset)
            if (
                not (isinstance(dataset[0], dict | BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths
        self.generator = generator

        self.mega_batch_mult = mega_batch_mult
        self.indices_order = indices_order
        self.grouped_indices = grouped_indices
        self.batch_total_length = batch_total_length
        self.epochs = []
        self.current_epoch_index = 0
        self.get_epoch(0)

    def _generate_epoch(self) -> list[list[int]]:
        """Generate a new epoch."""
        epoch = get_length_grouped_batches(
            self.lengths,
            self.batch_size,
            generator=self.generator,
            mega_batch_mult=self.mega_batch_mult,
            indices_order=self.indices_order,
            grouped_indices=self.grouped_indices,
            batch_total_length=self.batch_total_length,
        )
        utilization_ratio = np.mean(
            [sum([self.lengths[i] for i in b]) / self.batch_total_length for b in epoch]
        ).item()
        logger.info(
            f"Generated epoch with {len(epoch)} batches, utilization ratio: {utilization_ratio:.2%}"
        )
        return epoch

    def pregenerate_enough_steps(self, total_steps: int) -> None:
        """Get the epoch with the given index."""
        while sum(len(e) for e in self.epochs) <= total_steps:
            self.epochs.append(self._generate_epoch())

    def get_epoch(self, epoch_index: int) -> None:
        """Get the epoch with the given index."""
        while len(self.epochs) <= epoch_index:
            self.epochs.append(self._generate_epoch())
        return self.epochs[epoch_index]

    def __len__(self) -> int:
        """Get the length of the sampler."""
        return len(self.epochs[self.current_epoch_index])

    def __iter__(self) -> Iterator[list[int]]:
        """Get iterator with shuffled indices."""
        epoch = self.get_epoch(self.current_epoch_index)
        self.current_epoch_index += 1
        return iter(epoch)
