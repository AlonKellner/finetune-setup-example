"""A torch dataset wrapper that caches audio items to flac files.

The rest of the metadata is cached as an in memory dict and a parquet file.
"""

import concurrent.futures
import os
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torchaudio as ta
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm


class FlacDataset(TorchDataset):
    """A wrapping dataset for caching audio as local flac files."""

    def __init__(
        self,
        inner_dataset: HFDataset | TorchDataset,
        cache_path: str | Path,
        sample_rate: int,
        metadata: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self._inner_dataset = inner_dataset
        self.cache_path = Path(cache_path)
        self.metadata_path = self.cache_path / "metadata.parquet"
        if metadata is None:
            metadata = self.load_metadata()
        self.metadata = metadata
        self.sample_rate = sample_rate

    def complete_metadata(self) -> None:
        """Make sure that the metadata is complete."""
        if len(self) == 0:
            self.set_metadata(self.load_metadata())
        if len(self.metadata) < len(self):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2 * min(32, os.cpu_count() + 4)  # type: ignore
            ) as executor:
                for _ in tqdm(
                    executor.map(lambda i: self.validate_item(i), range(len(self))),
                    total=len(self),
                ):
                    pass
                self.save_metadata()

    def validate_item(self, index: int) -> None:
        """Validate item."""
        item = self[index]
        for _ in range(3):
            if len(item["input_values"]) == item["length"]:
                return

            Path(item["file_path"]).unlink(missing_ok=True)
            item = self[index]
        assert len(item["input_values"]) == item["length"], (
            f"Length mismatch with index [{index}]"
        )

    def save_metadata(self) -> None:
        """Save the metadata as a parquet."""
        pl.from_dicts(
            [
                dict(i=i, **self.metadata[i])
                for i in range(len(self))
                if i in self.metadata
            ]
        ).sort(by="i").write_parquet(self.metadata_path)

    def __getattr__(self, name: str) -> Any:
        """Delegate to the inner if it has the attribute."""
        result = getattr(self._inner_dataset, name)
        if issubclass(type(result), HFDataset) or issubclass(
            type(result), TorchDataset
        ):
            result = FlacDataset(
                result, self.cache_path, self.sample_rate, self.metadata
            )
        return result

    def load_metadata(self) -> dict[int, dict[str, Any]]:
        """Load metadata from a file."""
        if self.metadata_path.exists():
            df = pl.read_parquet(self.metadata_path).sort(by="i")
            return {
                row["i"]: {k: v for k, v in row.items() if k != "i"}
                for row in df.to_dicts()
            }
        else:
            return dict()

    def set_metadata(self, metadata: dict[int, dict[str, Any]]) -> None:
        """Set the current metadata value."""
        self.metadata = metadata

    @property
    def column_names(self) -> list[str]:
        """A property supported by HF datasets."""
        if len(self.metadata) == 0:
            self[0]
        return [*next(iter(self.metadata.values())).keys(), "input_values"]

    def __len__(self) -> int:
        """Propegates the length of the inner dataset."""
        return len(self._inner_dataset)  # type: ignore

    def __getitems__(self, keys: list) -> list:
        """Can be used to get a batch using a list of integers indices."""
        return [self[k] for k in keys]

    def __getitem__(self, index: int | str) -> dict[str, Any] | Any:
        """Return the item corresponding to the index while caching both metadata and audio to files."""
        if isinstance(index, str):
            if index in self.column_names:
                self.complete_metadata()
                return [self.metadata[i][index] for i in range(len(self.metadata))]
            return self._inner_dataset[index]

        item = None

        index = index % len(self)

        padded_index = str(index).zfill(len(str(len(self._inner_dataset))))  # type: ignore
        flac_path = self.cache_path / f"{padded_index}.flac"
        if not flac_path.exists():
            if item is None:
                item = self._inner_dataset[index]
            samples = item["input_values"]
            self._save_flac(flac_path, samples)

        samples = self._load_flac(flac_path)

        if index in self.metadata:
            item_metadata = self.metadata[index]
        else:
            if item is None:
                item = self._inner_dataset[index]
            item_metadata = {k: v for k, v in item.items() if k != "input_values"}
            item_metadata["indices"] = index
            item_metadata["file_paths"] = str(flac_path)
            item_metadata["file_sizes"] = flac_path.stat().st_size
            self.metadata[index] = item_metadata

        item = dict(
            input_values=samples,
            **item_metadata,
        )
        return item

    def _save_flac(self, flac_path: Path, samples: list[int]) -> None:
        _samples = torch.tensor(samples)
        _samples = _samples - _samples.min()
        _samples = _samples / _samples.max()
        _samples = (_samples * (2**31 - 2**24)).to(torch.int32)
        ta.save(
            str(flac_path.absolute()),
            _samples[None, :],
            self.sample_rate,
            format="flac",
            bits_per_sample=24,
            encoding="PCM_S",
        )

    def _load_flac(self, flac_path: Path) -> list[int]:
        samples, sr = ta.load(flac_path.absolute(), normalize=False, format="flac")
        assert sr == self.sample_rate
        assert samples.shape[0] == 1
        samples = samples[0, :]
        samples = (samples / (2**30 - 2**24)).to(torch.float32)
        samples = samples - samples.mean()
        samples = samples / samples.std()
        return samples.tolist()
