"""A torch dataset that saves groups of items as tars in S3."""

import concurrent.futures
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm

from ..tar_s3 import TarS3Syncer
from .compressed import FileDataset


class TarS3Dataset(TorchDataset):
    """A wrapping dataset for caching files to s3."""

    def __init__(
        self,
        inner_dataset: FileDataset,
        cache_path: str | Path,
        syncer: TarS3Syncer,
        cache_bucket: str,
        indices_order: list[int],
        max_tar_bytes: int,
        sync_interval: int = 2,
        groups_per_sync: int = 6,
        should_clean_groups: bool = False,
        should_sync_previous: bool = False,
        sync_on_start: bool = False,
    ) -> None:
        self._inner_dataset = inner_dataset
        self.cache_path = Path(cache_path)
        self.syncer = syncer
        self.cache_bucket = cache_bucket
        self.max_tar_bytes = max_tar_bytes
        self.sync_interval = sync_interval
        self.groups_per_sync = groups_per_sync
        self.should_clean_groups = should_clean_groups
        self.should_sync_previous = should_sync_previous
        self.sync_on_start = sync_on_start

        if not self.syncer._bucket_exists(self.cache_bucket):
            self.syncer._create_bucket(self.cache_bucket)

        for _ in tqdm(list(range(1))):
            metadata_path = self.sync_metadata()
            print(f"Metadata synced to {metadata_path}")
        inner_dataset.complete_metadata()
        for _ in tqdm(list(range(1))):
            self.sync_metadata()

        lengths = {i: self._inner_dataset.metadata[i]["length"] for i in indices_order}  # type: ignore
        longest_index = max(indices_order, key=lambda i: lengths[i])
        indices_order.remove(longest_index)
        indices_order.insert(0, longest_index)

        self._indices_order = indices_order

        file_sizes = {i: v["file_sizes"] for i, v in inner_dataset.metadata.items()}
        self._file_sizes = file_sizes

        cum_sizes = np.array([file_sizes[i] for i in indices_order]).cumsum()
        i = 0
        prev_i = 0
        self.grouped_indices = []
        while (pos_sizes := ((cum_sizes - self.max_tar_bytes) > 0)).any().item():
            prev_i = i
            i = np.diff(pos_sizes).argmax().item() + 1
            cum_sizes -= cum_sizes[i - 1]

            self.grouped_indices.append(indices_order[prev_i:i])
        self.grouped_indices.append(indices_order[i:])

        self._indices_groups = {
            v: i for i, vals in enumerate(self.grouped_indices) for v in vals
        }
        self._indices_flac_paths = {
            i: self._inner_dataset.get_full_name(i) for i in indices_order
        }

        self.sync_indices = [
            group[0] for group in self.grouped_indices[::sync_interval]
        ]

        if sync_on_start:
            self.sync_all_groups()
        else:
            self.sync_multiple_groups(
                list(range(min(groups_per_sync, len(self.grouped_indices))))
            )

    def __getattr__(self, name: str) -> Any:
        """Delegate to the inner if it has the attribute."""
        result = getattr(self._inner_dataset, name)
        if issubclass(type(result), HFDataset) or issubclass(
            type(result), TorchDataset
        ):
            result = TarS3Dataset(
                result,
                self.cache_path,
                self.syncer,
                self.cache_bucket,
                self._indices_order,
                self.max_tar_bytes,
                self.sync_interval,
                self.groups_per_sync,
                self.should_clean_groups,
                self.should_sync_previous,
            )
        return result

    def group_exists(self, group: int) -> bool:
        """Check whether a group exists in s3 cache or not."""
        padded_group = str(group % len(self.grouped_indices)).zfill(
            len(str(len(self.grouped_indices)))
        )  # type: ignore
        return self.syncer._exists(padded_group, self.cache_bucket)

    def __len__(self) -> int:
        """Propegates the length of the inner dataset."""
        return len(self._inner_dataset)  # type: ignore

    def __getitems__(self, keys: list) -> list:
        """Can be used to get a batch using a list of integers indices."""
        return [self[k] for k in keys]

    def __getitem__(self, index: int | str) -> dict[str, Any] | Any:
        """Return the item corresponding to the index while caching both metadata and audio to files."""
        if isinstance(index, str):
            return self._inner_dataset[index]

        index = index % len(self)

        current_group = self._indices_groups[index]

        if index in self.sync_indices:
            self.start_sync(current_group)

        return self._inner_dataset[index]

    def start_sync(self, current_group: int) -> None:
        """Start a parallel sync using threading."""
        thread = threading.Thread(
            target=self.sync_adjacent_groups, args=(current_group,)
        )
        thread.start()

    def sync_adjacent_groups(self, current_group: int) -> None:
        """Sync adjacent groups."""
        for i in range(0, current_group + self.groups_per_sync):
            self.sync_group(i)
        if self.should_sync_previous:
            for i in range(current_group - self.groups_per_sync, 0):
                self.sync_group(i)
        if self.should_clean_groups:
            for i in range(
                current_group - 2 * self.groups_per_sync,
                current_group - self.groups_per_sync,
            ):
                self.clean_group(i)

    def sync_group(self, group: int) -> None:
        """Sync the group of local files with s3 tar."""
        group = group % len(self.grouped_indices)
        padded_group = str(group).zfill(len(str(len(self.grouped_indices))))  # type: ignore
        group_flac_paths = [
            self._indices_flac_paths[i] for i in self.grouped_indices[group]
        ]
        self.syncer.sync_multiple_files(
            group_flac_paths, padded_group, self.cache_bucket, self.cache_path
        )

    def clean_group(self, group: int) -> None:
        """Sync the group of local files with s3 tar."""
        group = group % len(self.grouped_indices)
        group_flac_paths = [
            self._indices_flac_paths[i] for i in self.grouped_indices[group]
        ]
        for p in group_flac_paths:
            p.unlink(missing_ok=True)

    def sync_all_groups(self) -> None:
        """Sync all groups."""
        self.sync_multiple_groups(list(range(len(self.grouped_indices))))

    def sync_multiple_groups(self, groups: list[int]) -> None:
        """Sync multiple groups."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2 * min(32, os.cpu_count() + 4)  # type: ignore
        ) as executor:
            for _ in tqdm(
                executor.map(
                    self.sync_group,
                    groups,
                ),
                total=len(groups),
            ):
                pass

    def sync_metadata(self) -> Path:
        """Sync the metadata file with s3."""
        return self.syncer.sync_file(
            self.cache_path / "metadata.parquet", self.cache_bucket, self.cache_path
        )
