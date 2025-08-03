"""Common voice loading utilities."""

import contextlib
import random
import re
from pathlib import Path
from typing import Any, Literal

import dill
import iso639
import numpy as np
import uroman
from datasets import Audio, concatenate_datasets, load_dataset
from datasets import Dataset as HFDataset
from iso639 import LanguageNotFoundError
from torch.utils.data import Dataset as TorchDataset
from transformers.utils import PaddingStrategy

from ..custom_datasets import prepare_cached_dataset
from ..custom_datasets.lazy import LazyDataset
from ..custom_datasets.resized import ResizedDataset
from ..custom_wav2vec2.processor import CustomProcessorMixin
from ..tar_s3 import TarS3Syncer
from .languages import LANGUAGE_NAMES, LANGUAGES

Batch = dict[str, Any]


class Uromanizer:
    """A uromanizer class."""

    def __init__(self) -> None:
        self.uroman = uroman.Uroman()
        self.chars_to_remove_regex = r"[`,?\.!\-;:\"“%‘”�()…’]"  # noqa: RUF001

    def uromanize(self, batch: Batch) -> Batch:
        """Uromanize text."""
        iso3_code = batch["iso3_code"]
        assert (iso3_code is None) or (isinstance(iso3_code, str))
        clean_string = re.sub(self.chars_to_remove_regex, "", batch["sentence"]).lower()
        batch["sentence"] = self.uroman.romanize_string(clean_string, lcode=iso3_code)
        return batch


def infer_iso3(iso1_code: str, language_name: str) -> str | None:
    """Infer ISO3 code from ISO1 code."""
    _iso1_code = iso1_code
    _language_name = language_name
    try:
        return iso639.Language.match(_iso1_code).part3
    except LanguageNotFoundError:
        pass
    try:
        return iso639.Language.match(_language_name).part3
    except LanguageNotFoundError:
        pass
    while "-" in _iso1_code:
        _iso1_code = "-".join(_iso1_code.split("-")[:-1])
        try:
            return iso639.Language.match(_iso1_code).part3
        except LanguageNotFoundError:
            pass
    while " " in _language_name:
        _language_name = " ".join(_language_name.split(" ")[:-1])
        try:
            return iso639.Language.match(_language_name).part3
        except LanguageNotFoundError:
            pass
    print(f"WARNING: Language not found for {iso1_code} ({language_name})")
    return None


class LazyLoader:
    """A lazy loader for datasets."""

    def __init__(
        self,
        processor: CustomProcessorMixin,
        sample_rate: int,
        split: str,
        data_seed: int,
        features_name: str,
        cpu_count: int,
        total_languages: int | None = None,
    ) -> None:
        self.processor = processor
        self.sample_rate = sample_rate
        self.split = split
        self.data_seed = data_seed
        self.features_name = features_name
        self.total_languages = total_languages
        self.cpu_count = cpu_count
        self.common_voice_split = None
        self.meta_common_voice_split = None
        self.uromanizer = Uromanizer()

    def load_common_voice_for_wav2vec2(self) -> HFDataset:
        """Load a split of common voice, adapted for wav2vec2."""
        if self.common_voice_split is None:
            self.common_voice_split = self._load_common_voice_for_wav2vec2()
        return self.common_voice_split

    def prepare_dataset(self, batch: Batch) -> Batch:
        """Prepare dataset."""
        for _ in range(3):
            with contextlib.suppress(Exception):
                return self._prepare_dataset(batch)

        return self._prepare_dataset(batch)

    def _prepare_dataset(self, batch: Batch) -> Batch:
        """Prepare dataset."""
        audio = [a["array"] for a in batch["audio"]]
        sample_lengths = [len(a) for a in audio]
        sorted_idx = np.argsort(sample_lengths)
        reverse_idx = np.argsort(sorted_idx)
        sorted_audio = [audio[i] for i in sorted_idx]
        bulk_size = 32
        sorted_features = []
        for i in range(0, len(sorted_audio), bulk_size):
            sorted_features_bulk = self.processor(
                sorted_audio[i : i + bulk_size],
                sampling_rate=self.sample_rate,
                padding=PaddingStrategy.DO_NOT_PAD,
            )
            sorted_features.extend(sorted_features_bulk)
        features = [sorted_features[i] for i in reverse_idx]
        batch[self.features_name] = [f[self.features_name][0] for f in features]
        batch["length"] = [len(item) for item in batch[self.features_name]]
        batch["seconds"] = [s / self.sample_rate for s in sample_lengths]
        return batch

    def _load_common_voice_part(self, iso1_code: str) -> HFDataset | None:
        """Load a part of common voice."""
        language_name = LANGUAGE_NAMES[iso1_code]
        iso3_code = infer_iso3(iso1_code, language_name)
        language_id = f"{iso1_code}:{iso3_code} ({language_name})"

        print(f"Loading {language_id} common voice {self.split}...")
        try:
            dataset = load_dataset(
                "fsicoli/common_voice_22_0",
                iso1_code,
                split=self.split,
                token=True,
                trust_remote_code=True,
                num_proc=2 * min(32, self.cpu_count + 4),
            )
            blobs_path = Path(
                "~/.cache/huggingface/hub/datasets--fsicoli--common_voice_22_0/blobs"
            ).expanduser()
            for blob in blobs_path.iterdir():
                blob.unlink()
        except ValueError:
            print("WARNING: Dataset not found for", language_id)
            return None

        def add_iso3_code(batch: Batch) -> Batch:
            """Add ISO3 code to the batch."""
            batch["iso3_code"] = iso3_code
            return batch

        dataset = dataset.map(add_iso3_code)
        size = len(dataset)  # type: ignore
        print(f"{language_id} common voice {self.split} loaded with {size} samples.")
        return dataset  # type: ignore

    def _load_common_voice_for_wav2vec2(self) -> HFDataset:
        """Load a split of common voice, adapted for wav2vec2."""
        languages_to_load = LANGUAGES
        if self.total_languages is not None:
            languages_to_load = LANGUAGES[: self.total_languages]
        common_voice_split_parts = [
            self._load_common_voice_part(language) for language in languages_to_load
        ]
        common_voice_split_parts = [
            p for p in common_voice_split_parts if p is not None
        ]
        common_voice_split = concatenate_datasets(common_voice_split_parts)
        assert isinstance(common_voice_split, HFDataset)

        common_voice_split = common_voice_split.remove_columns(
            [
                "accent",
                "age",
                "client_id",
                "down_votes",
                "gender",
                "locale",
                "segment",
                "up_votes",
                "variant",
            ]
        )

        dill.dumps(self.uromanizer.uromanize)
        common_voice_split = common_voice_split.map(
            self.uromanizer.uromanize, num_proc=2 * min(32, self.cpu_count + 4)
        )

        common_voice_split = common_voice_split.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate)
        )

        rand_int = random.randint(0, len(common_voice_split) - 1)

        sentence = common_voice_split[rand_int]["sentence"]

        print(common_voice_split.column_names)
        dill.dumps(self.prepare_dataset)
        common_voice_split = common_voice_split.map(
            self.prepare_dataset,
            remove_columns=[
                c
                for c in common_voice_split.column_names
                if c not in ["sentence", "iso3_code"]
            ],
            batched=True,
            batch_size=64,
            num_proc=self.cpu_count // 3,
        )
        print(common_voice_split.column_names)

        print("Target text:", sentence)

        common_voice_split = common_voice_split.shuffle(seed=self.data_seed)
        return common_voice_split

    def load_meta_common_voice_for_wav2vec2(self) -> HFDataset:
        """Load a split of common voice metadata, adapted for wav2vec2."""
        if self.meta_common_voice_split is None:
            common_voice_split = self.load_common_voice_for_wav2vec2()
            self.meta_common_voice_split = common_voice_split.select_columns(
                [c for c in common_voice_split.column_names if c != self.features_name]  # type: ignore
            )
        return self.meta_common_voice_split


def create_cached_common_voice_split(
    data_seed: int,
    sample_rate: int,
    general_name: str,
    split_size: int | None,
    split_limit: int | None,
    processor: CustomProcessorMixin,
    syncer: TarS3Syncer,
    split: str,
    sync_all_on_start: bool,
    should_sync_previous: bool,
    should_clean_groups: bool,
    should_clean_validate: bool,
    features_name: str,
    total_languages: int | None,
    cpu_count: int,
    architecture: Literal["wav2vec2", "w2v-bert2"],
) -> tuple[TorchDataset, list[list[int]]]:
    """Create a common voice split with caching."""
    cache_path = Path(
        f"./.app_cache/{general_name}/{total_languages or 'all'}/data/{architecture}/{data_seed}/{split}/"
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_bucket = (
        f"{general_name}-{total_languages or 'all'}-{architecture}-{data_seed}-{split}"
    )

    dataset_metadata_path = cache_path / "dataset_metadata.json"
    if not syncer._bucket_exists(cache_bucket):
        syncer._create_bucket(cache_bucket)
    syncer.sync_file(dataset_metadata_path, cache_bucket)

    loader = LazyLoader(
        processor=processor,
        sample_rate=sample_rate,
        split=split,
        data_seed=data_seed,
        features_name=features_name,
        total_languages=total_languages,
        cpu_count=cpu_count,
    )

    common_voice_split = LazyDataset(
        loader.load_common_voice_for_wav2vec2, dataset_metadata_path
    )
    meta_common_voice_split = LazyDataset(
        loader.load_meta_common_voice_for_wav2vec2, dataset_metadata_path
    )
    syncer.sync_file(dataset_metadata_path, cache_bucket)

    if split_size is not None:
        common_voice_split = ResizedDataset(common_voice_split, split_size)
        meta_common_voice_split = ResizedDataset(meta_common_voice_split, split_size)
    common_voice_split = prepare_cached_dataset(
        None,
        common_voice_split,
        meta_common_voice_split,
        sample_rate,
        cache_path,
        cache_bucket,
        syncer,
        sync_all_on_start=sync_all_on_start,
        should_sync_previous=should_sync_previous,
        should_clean_groups=should_clean_groups,
        should_clean_validate=should_clean_validate,
        cpu_count=cpu_count,
        features_name=features_name,
        architecture=architecture,
    )
    total_seconds = sum(
        common_voice_split._inner_dataset.metadata[i]["seconds"]
        for i in range(len(common_voice_split))
    )
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    total_days = total_hours / 24
    total_time_str = f"/{total_days:.2f} days / {total_hours:.2f} hours / {total_minutes:.2f} minutes / {total_seconds:.2f} seconds"
    print(f"Total speech time in {split}: {total_time_str}")
    grouped_indices = common_voice_split.grouped_indices
    if not processor.sp_model_path.exists():
        processor.train_bpe_tokenizer([s for s in common_voice_split["sentence"]])

    tokenizer = processor.convert_tokenizer_to_bpe()
    common_voice_split._inner_dataset.tokenizer = tokenizer

    if split_limit is not None:
        common_voice_split = ResizedDataset(common_voice_split, split_limit)

    item = common_voice_split[0]
    print("=== Index 0 ===")
    print("Keys:\t", item.keys())
    print("Sentence:\t", item["sentence"])
    print(
        "Labels:\t",
        [tokenizer._convert_id_to_token(li) for li in item["labels"]["input_ids"]],
    )
    print("Decoded 1:\t", tokenizer.decode(item["labels"]["input_ids"]))
    print(
        "Decoded 2:\t",
        tokenizer.decode([ii for i in item["labels"]["input_ids"] for ii in [i] * 3]),
    )
    for i in range(10):
        item = common_voice_split[0]
        print(
            f"Labels [{i}]:\t",
            [tokenizer._convert_id_to_token(li) for li in item["labels"]["input_ids"]],
        )
    return common_voice_split, grouped_indices
