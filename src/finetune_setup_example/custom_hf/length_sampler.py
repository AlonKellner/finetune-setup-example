"""A custom version of the `transformers` length grouped sampler."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch import Generator
from transformers.trainer_pt_utils import LengthGroupedSampler


def get_length_grouped_indices_shuffled(
    lengths: list[int],
    batch_size: int,
    mega_batch_mult: int | None = None,
    generator: Generator | None = None,
    indices_order: list[int] | None = None,
    grouped_indices: list[list[int]] | None = None,
    batch_total_length: int | None = None,
) -> list[int]:
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

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.

    indices, grouped_indices = shuffle_indices(
        lengths, generator, indices_order, grouped_indices
    )
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [
        indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)
    ]

    # Get the biggest batch first.
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

    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=((mega_i % 2) == 0))
        for mega_i, megabatch in enumerate(megabatches)
    ]

    indices = [i for megabatch in megabatches for i in megabatch]

    if batch_total_length is None:
        cum_sizes = np.array([1 for _ in indices]).cumsum()
        batch_total = batch_size
    else:
        cum_sizes = np.array([lengths[i] for i in indices]).cumsum()
        batch_total = batch_total_length
    i = 0
    prev_i = 0
    batches = []
    while (pos_sizes := ((cum_sizes := cum_sizes - batch_total) > 0)).any().item():
        prev_i = i
        i = np.diff(pos_sizes).argmax().item()

        batches.append(indices[prev_i:i])
    batches.append(indices[i:])

    first_batch = batches[0][0]
    batches = [batches[0][1:], *batches[1:]]

    batches = [
        [
            megabatch[i]
            for i in torch.randperm(len(megabatch), generator=generator).tolist()
        ]
        for megabatch in batches
    ]
    batches = [[first_batch, *batches[0]], *batches[1:]]

    indices = [i for megabatch in batches for batch in megabatch for i in batch]
    if indices_order is not None:
        assert len(indices) == len(indices_order)
        assert len(set(indices)) == len(indices_order)
        assert max(indices) == len(indices_order) - 1
    else:
        assert len(indices) == len(lengths)
        assert len(set(indices)) == len(lengths)
        assert max(indices) == len(lengths) - 1
    assert min(indices) == 0
    return indices


def shuffle_indices(
    lengths: list[int],
    generator: Generator | None = None,
    indices_order: list[int] | None = None,
    grouped_indices: list[list[int]] | None = None,
) -> tuple[list[int], list[list[int]] | None]:
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
        indices = torch.randperm(len(indices_order), generator=generator).tolist()
    else:
        indices = torch.randperm(len(lengths), generator=generator).tolist()
    return indices, grouped_indices


class CustomLengthGroupedSampler(LengthGroupedSampler):
    """Custom sampler for random order."""

    def __init__(
        self,
        *args: Any,
        mega_batch_mult: int | None = None,
        indices_order: list[int] | None = None,
        grouped_indices: list[list[int]] | None = None,
        batch_total_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mega_batch_mult = mega_batch_mult
        self.indices_order = indices_order
        self.grouped_indices = grouped_indices
        self.batch_total_length = batch_total_length

    def __len__(self) -> int:
        """Get the length of the sampler."""
        if self.indices_order is not None:
            return len(self.indices_order)
        else:
            return len(self.lengths)

    def __iter__(self) -> Iterator[int]:
        """Get iterator with shuffled indices."""
        indices = get_length_grouped_indices_shuffled(
            self.lengths,
            self.batch_size,
            generator=self.generator,
            mega_batch_mult=self.mega_batch_mult,
            indices_order=self.indices_order,
            grouped_indices=self.grouped_indices,
            batch_total_length=self.batch_total_length,
        )
        return iter(indices)
