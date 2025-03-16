"""A torch dataset wrapper that changes it's size to a custom size."""

from typing import Any

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset


class ResizedDataset(TorchDataset):
    """A wrapping dataset for caching files to s3."""

    def __init__(self, inner_dataset: HFDataset | TorchDataset, size: int) -> None:
        self._inner_dataset = inner_dataset
        self._size = size

    def __getattr__(self, name: str) -> Any:
        """Delegate to the inner if it has the attribute."""
        result = getattr(self._inner_dataset, name)
        if issubclass(type(result), HFDataset) or issubclass(
            type(result), TorchDataset
        ):
            result = ResizedDataset(result, self._size)
        elif self._is_resizable(result):
            result = self._resize(result)
        return result

    def __len__(self) -> int:
        """Return the size as specified."""
        return self._size

    @property
    def total_length(self) -> int:
        """Return the size as specified."""
        return self._size

    def __getitems__(self, keys: list) -> list:
        """Can be used to get a batch using a list of integers indices."""
        return [self[k] for k in keys]

    def __getitem__(self, index: int | str) -> dict[str, Any] | Any:
        """Return the item corresponding to the index while caching both metadata and audio to files."""
        if isinstance(index, str):
            result = self._inner_dataset[index]
            if self._is_resizable(result):  # type: ignore
                result = self._resize(result)
            return result

        if index > self._size:
            raise IndexError()
        inner_index = index % len(self._inner_dataset)  # type: ignore
        return self._inner_dataset[inner_index]

    def _is_resizable(self, x: Any) -> bool:
        return isinstance(x, list) and (len(x) == len(self._inner_dataset))  # type: ignore

    def _resize(self, values: list) -> list:
        """Resize a list to the custom size."""
        return [values[i % len(values)] for i in range(self._size)]
