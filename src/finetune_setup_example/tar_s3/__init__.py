"""Utilities for syncing files with S3 tar archives."""

import tarfile
import tempfile
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from types_boto3_s3.client import S3Client


class TarS3Syncer:
    """A wrapping dataset for caching files to s3."""

    def __init__(
        self,
        s3_client: S3Client,
        s3_client_v2: S3Client,
    ) -> None:
        self.s3_client = s3_client
        self.s3_client_v2 = s3_client_v2

    def __getstate__(self) -> dict[str, Any]:
        """Get stateless state."""
        return dict()

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

    def _exists(self, name: str, bucket: str) -> bool:
        try:
            head_metadata = self.s3_client.head_object(
                Bucket=bucket, Key=f"{name}.tar.gz"
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

    def _upload(self, files: list[Path], name: str, bucket: str) -> None:
        if self._exists(name, bucket):
            return
        with tempfile.TemporaryFile() as f:
            with tarfile.open(fileobj=f, mode="w:gz") as tar:
                for file in files:
                    if file.exists():
                        tar.add(str(file.absolute()), arcname=file.name)
                    else:
                        print(f"WARNING: {file} does not exist and will be skipped.")
            f.seek(0)
            self.s3_client_v2.upload_fileobj(
                Fileobj=f, Bucket=bucket, Key=f"{name}.tar.gz"
            )

    def _download(self, name: str, bucket: str, outpath: Path) -> None:
        with tempfile.TemporaryFile() as f:
            print(f"{name}.tar.gz")
            self.s3_client.download_fileobj(
                Bucket=bucket, Key=f"{name}.tar.gz", Fileobj=f
            )
            f.seek(0)
            with tarfile.open(fileobj=f, mode="r:gz") as tar:
                tar.extractall(path=str(outpath.absolute()), filter="data")

    def sync_multiple_files(
        self, files: list[Path], title: str, bucket: str, outpath: Path
    ) -> None:
        """Sync multiple files with s3 tar."""
        if all(p.exists() for p in files):
            self._upload(files, title, bucket)
        elif self._exists(title, bucket):
            self._download(title, bucket, outpath)
        else:
            print("WARNING: Files do not exist and cannot be synced:", title)

    def sync_file(self, file: Path, bucket: str, outpath: Path) -> Path:
        """Sync a single file."""
        if file.exists():
            self._upload([file], file.stem, bucket)
        elif self._exists(file.stem, bucket):
            self._download(file.stem, bucket, outpath)
        else:
            print(f"WARNING: A file does not exist and cannot be synced: {file}")
        return file
