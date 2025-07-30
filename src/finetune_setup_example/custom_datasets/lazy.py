"""A torch dataset wrapper that defers inner dataset creation to when accessed."""

import json
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset


class LazyDataset(TorchDataset):
    """A lazy dataset wrapper."""

    def __init__(
        self,
        inner_dataset_func: Callable[[], HFDataset | TorchDataset],
        dataset_metadata_path: Path,
        missing_attributes: list[str] | None = None,
    ) -> None:
        self._inner_dataset_func = inner_dataset_func
        self.missing_attributes = missing_attributes or ["set_epoch"]
        self._inner_dataset: HFDataset | TorchDataset | None = None
        self._lock = threading.Lock()
        if dataset_metadata_path.exists():
            with open(dataset_metadata_path) as f:
                metadata = json.load(f)
            self._size = metadata["size"]
        else:
            print("WARNING: LazyDataset resolving inner dataset to cache length")
            self._size = len(self._get_inner_dataset())
            with open(dataset_metadata_path, "w") as f:
                json.dump({"size": self._size}, f)

    def __getattr__(self, name: str) -> Any:
        """Delegate to the inner if it has the attribute."""
        if name in self.missing_attributes:
            raise AttributeError()
        if self._inner_dataset is None:
            print("WARNING: LazyDataset __getattr__ called for", name)

        return getattr(self._get_inner_dataset(), name)

    def __len__(self) -> int:
        """Return the size as specified."""
        return self._size

    @property
    def total_length(self) -> int:
        """Return the size as specified."""
        return self._size

    @property
    def total_dataset_length(self) -> int:
        """Return the size as specified."""
        return self._size

    def __getitems__(self, keys: list) -> list:
        """Can be used to get a batch using a list of integers indices."""
        return [self[k] for k in keys]

    def __getitem__(self, index: int | str) -> dict[str, Any] | Any:
        """Return the item corresponding to the index while caching both metadata and audio to files."""
        if self._inner_dataset is None:
            print("WARNING: LazyDataset __getitem__ called for", index)
        return self._get_inner_dataset()[index]

    def _get_inner_dataset(self) -> HFDataset | TorchDataset:
        """Ensure the inner dataset is loaded."""
        if self._inner_dataset is None:
            with self._lock:
                if self._inner_dataset is None:
                    print("WARNING: Resolving inner dataset...\nSTACKTRACE:")
                    traceback.print_stack()
                    self._inner_dataset = self._inner_dataset_func()
        return self._inner_dataset
