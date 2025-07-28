"""Common voice loading utilities."""

import random
import re
from pathlib import Path
from typing import Any, Literal

import dill
import iso639
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


class LazyLoader:
    """A lazy loader for datasets."""

    def __init__(
        self,
        processor: CustomProcessorMixin,
        sample_rate: int,
        split: str,
        data_seed: int,
        features_name: str,
    ) -> None:
        self.processor = processor
        self.sample_rate = sample_rate
        self.split = split
        self.data_seed = data_seed
        self.features_name = features_name
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
        audio = batch["audio"]
        batch[self.features_name] = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            padding=PaddingStrategy.DO_NOT_PAD,
        )[self.features_name][0]
        batch["length"] = len(batch[self.features_name])
        return batch

    def _load_common_voice_part(self, iso1_code: str) -> HFDataset:
        """Load a part of common voice."""
        language_name = LANGUAGE_NAMES[iso1_code]
        try:
            iso3_code = iso639.Language.match(iso1_code).part3
        except LanguageNotFoundError:
            try:
                iso3_code = iso639.Language.match(language_name).part3
            except LanguageNotFoundError:
                print(f"WARNING: Language not found for {iso1_code} ({language_name})")
                iso3_code = None
        language_id = f"{iso1_code}:{iso3_code} ({language_name})"

        print(f"Loading {language_id} common voice split...")
        dataset = load_dataset(
            "fsicoli/common_voice_22_0",
            iso1_code,
            split=self.split,
            token=True,
            trust_remote_code=True,
        )

        def add_iso3_code(batch: Batch) -> Batch:
            """Add ISO3 code to the batch."""
            batch["iso3_code"] = iso3_code
            return batch

        dataset = dataset.map(add_iso3_code)
        size = len(dataset)  # type: ignore
        print(f"{language_id} common voice split loaded with {size} samples.")
        return dataset  # type: ignore

    def _load_common_voice_for_wav2vec2(self) -> HFDataset:
        """Load a split of common voice, adapted for wav2vec2."""
        common_voice_split_parts = [
            self._load_common_voice_part(language) for language in LANGUAGES
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
            ]
        )

        dill.dumps(self.uromanizer.uromanize)
        common_voice_split = common_voice_split.map(self.uromanizer.uromanize)

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
                c for c in common_voice_split.column_names if c != "sentence"
            ],
        )
        print(common_voice_split.column_names)

        print("Target text:", sentence)
        norm_sentence = common_voice_split[rand_int]["sentence"]
        print("Norm Target text:", norm_sentence)
        ids = self.processor.tokenizer.sp_model.encode(
            norm_sentence,
            out_type=int,
            enable_sampling=False,
            nbest_size=-1,
        )
        labels = [self.processor.tokenizer.sp_model.id_to_piece(id) for id in ids]
        print("Target pcs:\t", labels)
        print("Target ids:\t", ids)
        for i in range(10):
            ids = self.processor(
                text=common_voice_split[rand_int]["sentence"]
            ).input_ids
            print(f"Processed Target ids [{i}]:\t", ids)

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
    sync_on_start: bool,
    features_name: str,
    architecture: Literal["wav2vec2", "w2v-bert2"],
) -> tuple[TorchDataset, list[list[int]]]:
    """Create a common voice split with caching."""
    cache_path = Path(f"./.app_cache/{data_seed}/{architecture}/{split}_set/")
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_bucket = f"{general_name}-cache-{data_seed}-{architecture}-{split}-set"

    dataset_metadata_path = cache_path / "dataset_metadata.json"
    syncer.sync_file(dataset_metadata_path, cache_bucket)

    loader = LazyLoader(
        processor=processor,
        sample_rate=sample_rate,
        split=split,
        data_seed=data_seed,
        features_name=features_name,
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
        sync_on_start=sync_on_start,
        features_name=features_name,
        architecture=architecture,
    )
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
