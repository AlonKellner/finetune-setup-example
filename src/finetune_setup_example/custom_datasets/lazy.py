"""A torch dataset wrapper that defers inner dataset creation to when accessed."""

import threading
from collections.abc import Callable
from typing import Any

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset


class LazyDataset(TorchDataset):
    """A lazy dataset wrapper."""

    def __init__(
        self, inner_dataset_func: Callable[[], HFDataset | TorchDataset], size: int
    ) -> None:
        self._inner_dataset_func = inner_dataset_func
        self._size = size
        self._inner_dataset: HFDataset | TorchDataset | None = None
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        """Delegate to the inner if it has the attribute."""
        _inner_dataset = self._get_inner_dataset()

        return getattr(_inner_dataset, name)

    def __len__(self) -> int:
        """Return the size as specified."""
        if self._inner_dataset is None:
            return self._size
        else:
            real_size = len(self._inner_dataset)  # type: ignore
            if real_size != self._size:
                print("WARNING! Lazy dataset sizes mismatch!")
            return real_size

    @property
    def total_length(self) -> int:
        """Return the size as specified."""
        return self._size

    def __getitems__(self, keys: list) -> list:
        """Can be used to get a batch using a list of integers indices."""
        return [self[k] for k in keys]

    def __getitem__(self, index: int | str) -> dict[str, Any] | Any:
        """Return the item corresponding to the index while caching both metadata and audio to files."""
        _inner_dataset = self._get_inner_dataset()

        return _inner_dataset[index]

    def _get_inner_dataset(self) -> HFDataset | TorchDataset:
        """Ensure the inner dataset is loaded."""
        if self._inner_dataset is None:
            with self._lock:
                if self._inner_dataset is None:
                    self._inner_dataset = self._inner_dataset_func()
        return self._inner_dataset
