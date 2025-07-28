"""Custom torch dataset wrappers."""

from pathlib import Path
from typing import Literal

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset

from ..custom_wav2vec2.tokenizer import BpeWav2Vec2CTCTokenizer
from ..tar_s3 import TarS3Syncer
from .compressed import FlacDataset, TifDataset
from .tar_s3 import TarS3Dataset

GB = 1_000_000_000


def prepare_cached_dataset(
    tokenizer: BpeWav2Vec2CTCTokenizer | None,
    dataset: TorchDataset | HFDataset,
    meta_dataset: TorchDataset | HFDataset,
    sample_rate: int,
    cache_path: Path,
    cache_bucket: str,
    syncer: TarS3Syncer,
    features_name: str,
    architecture: Literal["wav2vec2", "w2v-bert2"],
    tar_size_gb: float | int = 1,
    sync_interval: int = 2,
    groups_per_sync: int = 6,
    sync_on_start: bool = False,
) -> TarS3Dataset:
    """Wrap a dataset with S3 caching and prepare for the first training."""
    if architecture == "wav2vec2":
        dataset = FlacDataset(
            sample_rate=sample_rate,
            inner_dataset=dataset,
            inner_meta_dataset=meta_dataset,
            cache_path=cache_path,
            tokenizer=tokenizer,
            features_name=features_name,
        )
    elif architecture == "w2v-bert2":
        dataset = TifDataset(
            inner_dataset=dataset,
            inner_meta_dataset=meta_dataset,
            cache_path=cache_path,
            tokenizer=tokenizer,
            features_name=features_name,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    indices_order = list(range(len(dataset)))
    dataset = TarS3Dataset(
        dataset,
        cache_path,
        syncer,
        cache_bucket,
        indices_order,
        max_tar_bytes=int(tar_size_gb * GB),
        sync_interval=sync_interval,
        groups_per_sync=groups_per_sync,
        sync_on_start=sync_on_start,
    )
    return dataset
