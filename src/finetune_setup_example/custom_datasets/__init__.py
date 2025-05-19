"""Custom torch dataset wrappers."""

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from types_boto3_s3.client import S3Client

from .flac import FlacDataset
from .tar_s3 import TarS3Dataset

GB = 1_000_000_000


def prepare_cached_dataset(
    dataset: TorchDataset | HFDataset,
    sample_rate: int,
    cache_path: str,
    cache_bucket: str,
    s3_client: S3Client,
    s3_client_v2: S3Client,
    tar_size_gb: float | int = 1,
    sync_interval: int = 2,
    groups_per_sync: int = 6,
) -> TarS3Dataset:
    """Wrap a dataset with S3 caching and prepare for the first training."""
    dataset = FlacDataset(dataset, cache_path, sample_rate)
    indices_order = list(range(len(dataset)))
    dataset = TarS3Dataset(
        dataset,
        cache_path,
        s3_client,
        s3_client_v2,
        cache_bucket,
        indices_order,
        max_tar_bytes=int(tar_size_gb * GB),
        sync_interval=sync_interval,
        groups_per_sync=groups_per_sync,
    )
    # dataset.sync_all_groups()
    return dataset
