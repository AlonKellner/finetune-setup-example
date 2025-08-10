"""A torch dataset wrapper that changes it's size to a custom size."""

from typing import Any

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset


class RemappedDataset(TorchDataset):
    """A wrapping dataset for remapping and subsetting indices."""

    def __init__(
        self, inner_dataset: HFDataset | TorchDataset, remapping: list[int]
    ) -> None:
        self._inner_dataset = inner_dataset
        self._remapping = remapping

    def __getattr__(self, name: str) -> Any:
        """Delegate to the inner if it has the attribute."""
        if name not in [
            "grouped_indices",
            "indices_order",
            "metadata",
            "column_names",
            "total_length",
            "total_dataset_length",
            "set_tokenizer",
        ]:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        result = getattr(self._inner_dataset, name)
        if issubclass(type(result), HFDataset) or issubclass(
            type(result), TorchDataset
        ):
            result = RemappedDataset(result, self._remapping)
        elif self._is_remappable(result):
            result = self._remap(result)
        return result

    def __len__(self) -> int:
        """Return the size as specified."""
        return len(self._remapping)

    def __getitems__(self, keys: list) -> list:
        """Can be used to get a batch using a list of integers indices."""
        return [self[k] for k in keys]

    def __getitem__(self, index: int | str) -> dict[str, Any] | Any:
        """Return the item corresponding to the index while caching both metadata and audio to files."""
        if isinstance(index, str):
            result = self._inner_dataset[index]
            if self._is_remappable(result):  # type: ignore
                result = self._remap(result)
            return result

        if index > len(self):
            raise IndexError()
        inner_index = self._remapping[index]
        return self._inner_dataset[inner_index]

    def _is_remappable(self, x: Any) -> bool:
        return isinstance(x, list) and (len(x) == len(self._inner_dataset))  # type: ignore

    def _remap(self, values: list) -> list:
        """Resize a list to the custom size."""
        return [values[i] for i in self._remapping]
