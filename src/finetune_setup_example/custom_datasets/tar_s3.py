"""A torch dataset that saves groups of items as tars in S3."""

import concurrent.futures
import os
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from botocore.exceptions import ClientError
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm
from types_boto3_s3.client import S3Client

from .flac import FlacDataset


class TarS3Dataset(TorchDataset):
    """A wrapping dataset for caching files to s3."""

    def __init__(
        self,
        inner_dataset: FlacDataset,
        cache_path: str | Path,
        s3_client: S3Client,
        s3_client_v2: S3Client,
        cache_bucket: str,
        indices_order: list[int],
        max_tar_bytes: int,
        sync_interval: int = 2,
        groups_per_sync: int = 6,
        should_clean_groups: bool = False,
    ) -> None:
        self._inner_dataset = inner_dataset
        self.cache_path = Path(cache_path)
        self.s3_client = s3_client
        self.s3_client_v2 = s3_client_v2
        self.cache_bucket = cache_bucket
        self.max_tar_bytes = max_tar_bytes
        self.sync_interval = sync_interval
        self.groups_per_sync = groups_per_sync
        self.should_clean_groups = should_clean_groups

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
        self._indices_flac_paths = {i: self._get_flac_path(i) for i in indices_order}

        self.sync_indices = [group[0] for group in self.grouped_indices]

        if not self._bucket_exists(self.cache_bucket):
            self._create_bucket(self.cache_bucket)

        for _ in tqdm(list(range(1))):
            self.sync_group(0)

    def __getattr__(self, name: str) -> Any:
        """Delegate to the inner if it has the attribute."""
        result = getattr(self._inner_dataset, name)
        if issubclass(type(result), HFDataset) or issubclass(
            type(result), TorchDataset
        ):
            result = TarS3Dataset(
                result,
                self.cache_path,
                self.s3_client,
                self.s3_client_v2,
                self.cache_bucket,
                self._indices_order,
                self.max_tar_bytes,
                self.sync_interval,
                self.groups_per_sync,
                self.should_clean_groups,
            )
        return result

    def _bucket_exists(self, bucket: str) -> bool:
        try:
            head_metadata = self.s3_client.head_bucket(Bucket=bucket)[
                "ResponseMetadata"
            ]
            return (
                ("HTTPStatusCode" in head_metadata)
                and (head_metadata["HTTPStatusCode"] == 200)
                and ("HTTPHeaders" in head_metadata)
                and ("content-length" in head_metadata["HTTPHeaders"])
                and (int(head_metadata["HTTPHeaders"]["content-length"]) == 0)
            )
        except ClientError:
            return False

    def _create_bucket(self, bucket: str) -> None:
        self.s3_client.create_bucket(Bucket=bucket)

    def _exists(self, name: str) -> bool:
        try:
            head_metadata = self.s3_client.head_object(
                Bucket=self.cache_bucket, Key=f"{name}.tar.gz"
            )["ResponseMetadata"]
            return (
                ("HTTPStatusCode" in head_metadata)
                and (head_metadata["HTTPStatusCode"] == 200)
                and ("HTTPHeaders" in head_metadata)
                and ("content-length" in head_metadata["HTTPHeaders"])
                and (int(head_metadata["HTTPHeaders"]["content-length"]) > 0)
            )
        except ClientError:
            return False

    def group_exists(self, group: int) -> bool:
        """Check whether a group exists in s3 cache or not."""
        padded_group = str(group % len(self.grouped_indices)).zfill(
            len(str(len(self.grouped_indices)))
        )  # type: ignore
        return self._exists(padded_group)

    def _upload(self, files: list[Path], name: str) -> None:
        if self._exists(name):
            return
        with tempfile.TemporaryFile() as f:
            with tarfile.open(fileobj=f, mode="w:gz") as tar:
                for file in files:
                    if file.exists():
                        tar.add(str(file.absolute()), arcname=file.name)
                    else:
                        print(f"Warning: {file} does not exist and will be skipped.")
            f.seek(0)
            self.s3_client_v2.upload_fileobj(
                Fileobj=f, Bucket=self.cache_bucket, Key=f"{name}.tar.gz"
            )

    def _download(self, name: str) -> None:
        with tempfile.TemporaryFile() as f:
            self.s3_client.download_fileobj(
                Bucket=self.cache_bucket, Key=f"{name}.tar.gz", Fileobj=f
            )
            f.seek(0)
            with tarfile.open(fileobj=f, mode="r:gz") as tar:
                tar.extractall(path=str(self.cache_path.absolute()), filter="data")

    def __len__(self) -> int:
        """Propegates the length of the inner dataset."""
        return len(self._inner_dataset)  # type: ignore

    def __getitems__(self, keys: list) -> list:
        """Can be used to get a batch using a list of integers indices."""
        return [self[k] for k in keys]

    def _get_flac_path(self, index: int) -> Path:
        """Get the flac path of an index."""
        padded_index = str(index).zfill(len(str(len(self._inner_dataset))))  # type: ignore
        flac_path = self.cache_path / f"{padded_index}.flac"
        return flac_path

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
        for i in range(
            current_group - self.groups_per_sync, current_group + self.groups_per_sync
        ):
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
        if all(p.exists() for p in group_flac_paths):
            self._upload(group_flac_paths, padded_group)
        elif self._exists(padded_group):
            self._download(padded_group)
        else:
            print("WARNING: A group does not exist and cannot be synced:", padded_group)

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
        if any((not self.group_exists(i)) for i in range(len(self.grouped_indices))):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2 * min(32, os.cpu_count() + 4)  # type: ignore
            ) as executor:
                for _ in tqdm(
                    executor.map(
                        self.sync_group,
                        range(len(self.grouped_indices)),
                    ),
                    total=len(self.grouped_indices),
                ):
                    pass

    def sync_metadata(self) -> Path:
        """Sync the metadata file with s3."""
        return self.sync_file(self.cache_path / "metadata.parquet")

    def sync_file(self, file: Path) -> Path:
        """Sync a single file."""
        if file.exists():
            self._upload([file], file.stem)
        elif self._exists(file.stem):
            self._download(file.stem)
        else:
            print(f"WARNING: A file does not exist and cannot be synced: {file}")
        return file
