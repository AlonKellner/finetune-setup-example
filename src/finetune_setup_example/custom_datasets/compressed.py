"""A torch dataset wrapper that caches audio items to flac files.

The rest of the metadata is cached as an in memory dict and a parquet file.
"""

from __future__ import annotations

import concurrent.futures
import os
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import tifffile
import torch
import torchaudio as ta
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from ..custom_wav2vec2.tokenizer import BpeWav2Vec2CTCTokenizer


class FileDataset(ABC, TorchDataset):
    """An abstraction for file datasets."""

    def __init__(
        self,
        inner_dataset: HFDataset | TorchDataset,
        inner_meta_dataset: HFDataset | TorchDataset,
        cache_path: Path,
        features_name: str,
        tokenizer: BpeWav2Vec2CTCTokenizer | None,
        metadata: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self._inner_dataset = inner_dataset
        self._inner_meta_dataset = inner_meta_dataset
        self.cache_path = Path(cache_path)
        self.metadata_path = self.cache_path / "metadata.parquet"
        if metadata is None:
            metadata = self.load_metadata()
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.features_name = features_name

    def complete_metadata(self) -> None:
        """Make sure that the metadata is complete."""
        if len(self.metadata) == 0:
            self.set_metadata(self.load_metadata())
        if len(self.metadata) < len(self):
            print(
                f"WARNING: Metadata is incomplete ({len(self.metadata)}/{len(self)}), validating items...\nSTACKTRACE:"
            )
            traceback.print_stack()
            try:
                self.validate_item(0)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=2 * min(32, os.cpu_count() + 4)  # type: ignore
                ) as executor:
                    for _ in tqdm(
                        executor.map(lambda i: self.validate_item(i), range(len(self))),
                        total=len(self),
                    ):
                        pass
            finally:
                self.save_metadata()

    def validate_item(self, index: int) -> None:
        """Validate item."""
        item = self[index]
        for _ in range(3):
            if len(item[self.features_name]) == item["length"]:
                return

            Path(item["file_path"]).unlink(missing_ok=True)
            item = self[index]
        assert len(item[self.features_name]) == item["length"], (
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
        meta_result = getattr(self._inner_meta_dataset, name)
        if issubclass(type(result), HFDataset) or issubclass(
            type(result), TorchDataset
        ):
            result = self.re_init(result, meta_result)
        return result

    @abstractmethod
    def re_init(
        self,
        inner_dataset: TorchDataset | HFDataset,
        inner_meta_dataset: TorchDataset | HFDataset,
    ) -> FileDataset:
        """Recreate the dataset with the new inner ones."""
        pass

    def load_metadata(self) -> dict[int, dict[str, Any]]:
        """Load metadata from a file."""
        if self.metadata_path.exists():
            df = pl.read_parquet(self.metadata_path).sort(by="i")
            return {
                row["i"]: {k: v for k, v in row.items() if k != "i"}
                for row in df.to_dicts()
            }
        else:
            print(
                f"WARNING: Metadata file does not exist in: {self.metadata_path}\nCreating empty metadata."
            )
            return dict()

    def set_metadata(self, metadata: dict[int, dict[str, Any]]) -> None:
        """Set the current metadata value."""
        self.metadata = metadata

    @property
    def column_names(self) -> list[str]:
        """A property supported by HF datasets."""
        if len(self.metadata) == 0:
            self[0]
        return [
            *next(iter(self.metadata.values())).keys(),
            self.features_name,
            "labels",
        ]

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

        full_path = self.get_full_name(index)

        if not full_path.exists():
            if item is None:
                item = self._inner_dataset[index]  # type: ignore
            features = item[self.features_name]
            self.save_file(full_path, features)

        try:
            features = self.load_file(full_path)
        except Exception:
            full_path.unlink()
            if item is None:
                item = self._inner_dataset[index]  # type: ignore
            features = item[self.features_name]
            self.save_file(full_path, features)
            features = self.load_file(full_path)

        if index in self.metadata:
            item_metadata = self.metadata[index]
        else:
            if item is None:
                item = self._inner_meta_dataset[index]
            item_metadata = {k: v for k, v in item.items() if k != self.features_name}
            item_metadata["indices"] = index
            item_metadata["file_paths"] = str(full_path)
            item_metadata["file_sizes"] = full_path.stat().st_size
            self.metadata[index] = item_metadata

        item_metadata = item_metadata.copy()
        if self.tokenizer is not None:
            item_metadata["labels"] = self.tokenizer(item_metadata["sentence"])
        item = {
            self.features_name: features,
            **item_metadata,
        }
        return item

    def get_full_name(self, index: int) -> Path:
        """Convert index to full file name."""
        padded_index = str(index).zfill(len(str(len(self._inner_dataset))))  # type: ignore
        return self.cache_path / self.get_base_name(padded_index)

    @abstractmethod
    def get_base_name(self, padded_index: str) -> str:
        """Convert padded index to base file name."""
        pass

    def save_file(
        self, path: Path, features: list[int | float] | list[list[int | float]]
    ) -> None:
        """Save a file with retries."""
        for i in range(3):
            try:
                return self.raw_save_file(path, features)
            except Exception as e:
                print(f"Failed to save file {path} with error: {e}")
                path.unlink(missing_ok=True)
                time.sleep(0.1 * 2**i)  # Give the filesystem some time to recover

        return self.raw_save_file(path, features)

    @abstractmethod
    def raw_save_file(
        self, path: Path, features: list[int | float] | list[list[int | float]]
    ) -> None:
        """Save an actual file."""
        pass

    def load_file(self, path: Path) -> list[int]:
        """Load a file with retries."""
        for i in range(3):
            try:
                return self.raw_load_file(path)
            except Exception as e:
                print(f"Failed to load file {path} with error: {e}")
                time.sleep(0.1 * 2**i)

        return self.raw_load_file(path)

    @abstractmethod
    def raw_load_file(self, path: Path) -> list[int | float] | list[list[int | float]]:
        """Load an actual file."""
        pass


class FlacDataset(FileDataset):
    """A wrapping dataset for caching audio as local flac files."""

    def __init__(
        self,
        sample_rate: int,
        inner_dataset: HFDataset | TorchDataset,
        inner_meta_dataset: HFDataset | TorchDataset,
        cache_path: Path,
        features_name: str,
        tokenizer: BpeWav2Vec2CTCTokenizer | None,
        metadata: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        super().__init__(
            inner_dataset=inner_dataset,
            inner_meta_dataset=inner_meta_dataset,
            cache_path=cache_path,
            features_name=features_name,
            tokenizer=tokenizer,
            metadata=metadata,
        )

    def re_init(
        self,
        inner_dataset: TorchDataset | HFDataset,
        inner_meta_dataset: TorchDataset | HFDataset,
    ) -> FileDataset:
        """Recreate the dataset with the new inner ones."""
        return FlacDataset(
            sample_rate=self.sample_rate,
            inner_dataset=inner_dataset,
            inner_meta_dataset=inner_meta_dataset,
            cache_path=self.cache_path,
            features_name=self.features_name,
            tokenizer=self.tokenizer,
            metadata=self.metadata,
        )

    def get_base_name(self, padded_index: str) -> str:
        """Convert padded index to base file name."""
        return f"{padded_index}.flac"

    def raw_save_file(
        self, path: Path, features: list[int | float] | list[list[int | float]]
    ) -> None:
        """Save a flac."""
        if hasattr(features[0], "__len__"):
            if len(features) == 1:
                features = features[0]  # type: ignore
            elif len(features[0]) == 1:
                features = [s[0] for s in features]  # type: ignore
            else:
                raise ValueError(
                    f"Samples have too many dimensions! {(len(features), len(features[0]))}"
                )
        _samples = torch.tensor(features)
        _samples = _samples - _samples.min()
        _samples = _samples / _samples.max()
        _samples = (_samples * (2**31 - 2**24)).to(torch.int32)
        if len(_samples.shape) == 1:
            _samples = _samples[None, :]
        ta.save(
            str(path.absolute()),
            _samples,
            self.sample_rate,
            format="flac",
            bits_per_sample=24,
        )

    def raw_load_file(self, path: Path) -> list[int | float] | list[list[int | float]]:
        """Load a flac."""
        samples, sr = ta.load(path.absolute(), normalize=False, format="flac")
        assert sr == self.sample_rate
        assert samples.shape[0] == 1
        samples = samples[0, :]
        samples = (samples / (2**30 - 2**24)).to(torch.float32)
        samples = samples - samples.mean()
        samples = samples / samples.std()
        return samples.tolist()


class TifDataset(FileDataset):
    """A wrapping dataset for caching 2d data as local tif files."""

    def __init__(
        self,
        inner_dataset: HFDataset | TorchDataset,
        inner_meta_dataset: HFDataset | TorchDataset,
        cache_path: Path,
        features_name: str,
        tokenizer: BpeWav2Vec2CTCTokenizer | None,
        metadata: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            inner_dataset=inner_dataset,
            inner_meta_dataset=inner_meta_dataset,
            cache_path=cache_path,
            features_name=features_name,
            tokenizer=tokenizer,
            metadata=metadata,
        )

    def re_init(
        self,
        inner_dataset: TorchDataset | HFDataset,
        inner_meta_dataset: TorchDataset | HFDataset,
    ) -> FileDataset:
        """Recreate the dataset with the new inner ones."""
        return TifDataset(
            inner_dataset=inner_dataset,
            inner_meta_dataset=inner_meta_dataset,
            cache_path=self.cache_path,
            features_name=self.features_name,
            tokenizer=self.tokenizer,
            metadata=self.metadata,
        )

    def get_base_name(self, padded_index: str) -> str:
        """Convert padded index to base file name."""
        return f"{padded_index}.tif"

    def raw_save_file(
        self, path: Path, features: list[int | float] | list[list[int | float]]
    ) -> None:
        """Save a TIFF."""
        assert len(features) > 0
        assert isinstance(features[0], Sized)
        assert len(features[0]) > 0
        _image = np.array(features).T[None, :, :]
        _image = np.flip(_image, -2)
        tifffile.imwrite(str(path.absolute()), _image)

    def raw_load_file(self, path: Path) -> list[int | float] | list[list[int | float]]:
        """Load a TIFF."""
        _image = tifffile.imread(str(path.absolute()))
        _image = np.flip(_image, -2)
        return _image[0, :, :].T.tolist()
