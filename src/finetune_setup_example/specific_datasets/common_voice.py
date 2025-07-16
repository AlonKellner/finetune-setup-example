"""Common voice loading utilities."""

import random
import re
from pathlib import Path
from typing import Any

import sentencepiece as spm
import uroman
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from transformers.utils import PaddingStrategy

from ..custom_datasets import prepare_cached_dataset
from ..custom_datasets.lazy import LazyDataset
from ..custom_datasets.resized import ResizedDataset
from ..custom_wav2vec2.processor import CustomWav2Vec2Processor
from ..tar_s3 import TarS3Syncer

Batch = dict[str, Any]


class LazyLoader:
    """A lazy loader for datasets."""

    def __init__(
        self,
        processor: CustomWav2Vec2Processor,
        target_lang: str,
        sample_rate: int,
        split: str,
        data_seed: int,
        sp_dir: str,
        sp_vocab_size: int,
        sp_bpe_dropout: float,
        sp_extra_symbols: list[str] | None = None,
    ) -> None:
        self.processor = processor
        self.target_lang = target_lang
        self.sample_rate = sample_rate
        self.split = split
        self.data_seed = data_seed
        self.sp_dir = sp_dir
        self.sp_vocab_size = sp_vocab_size
        self.sp_bpe_dropout = sp_bpe_dropout
        self.sp_extra_symbols = [] if sp_extra_symbols is None else sp_extra_symbols
        self.common_voice_split = None
        self.meta_common_voice_split = None
        self.uroman = uroman.Uroman()
        self.chars_to_remove_regex = r"[`,?\.!\-;:\"“%‘”�()…’]"  # noqa: RUF001

    def uromanize(self, batch: Batch) -> Batch:
        """Uromanize text."""
        clean_string = re.sub(self.chars_to_remove_regex, "", batch["sentence"]).lower()
        batch["sentence"] = self.uroman.romanize_string(
            clean_string, lcode=self.target_lang
        )
        return batch

    def load_common_voice_for_wav2vec2(self) -> HFDataset:
        """Load a split of common voice 17, adapted for wav2vec2."""
        if self.common_voice_split is None:
            self.common_voice_split = self._load_common_voice_for_wav2vec2()
        return self.common_voice_split

    def prepare_dataset(self, batch: Batch) -> Batch:
        """Prepare dataset."""
        audio = batch["audio"]
        batch["input_values"] = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            padding=PaddingStrategy.DO_NOT_PAD,
        ).input_values[0]
        batch["length"] = len(batch["input_values"])
        return batch

    def _load_common_voice_for_wav2vec2(self) -> HFDataset:
        """Load a split of common voice 17, adapted for wav2vec2."""
        common_voice_split = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "tr",
            split=self.split,
            token=True,
            trust_remote_code=True,
        )
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

        common_voice_split = common_voice_split.map(self.uromanize)
        train_text = "\n".join([s for s in common_voice_split["sentence"]])
        train_text_path = f"{self.sp_dir}/train_text.txt"
        Path(self.sp_dir).mkdir(parents=True, exist_ok=True)
        with open(train_text_path, "w") as f:
            f.write(train_text)
        spm.SentencePieceTrainer.Train(
            input=train_text_path,
            model_prefix=f"{self.sp_dir}/spm",
            character_coverage=1.0,
            vocab_size=self.sp_vocab_size,
            user_defined_symbols=self.sp_extra_symbols,
            model_type="bpe",
            unk_id=0,
            pad_id=1,
            bos_id=2,
            eos_id=3,
        )
        if self.processor.can_create_bpe_tokenizer():
            self.processor.convert_tokenizer_to_bpe()

        common_voice_split = common_voice_split.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate)
        )

        rand_int = random.randint(0, len(common_voice_split) - 1)

        sentence = common_voice_split[rand_int]["sentence"]

        print(common_voice_split.column_names)
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
        """Load a split of common voice 17 metadata, adapted for wav2vec2."""
        if self.meta_common_voice_split is None:
            common_voice_split = self.load_common_voice_for_wav2vec2()
            self.meta_common_voice_split = common_voice_split.select_columns(
                [c for c in common_voice_split.column_names if c != "input_values"]  # type: ignore
            )
        return self.meta_common_voice_split


def create_cached_common_voice_split(
    data_seed: int,
    target_lang: str,
    sample_rate: int,
    target_hf_repo: str,
    raw_split_size: int,
    split_size: int,
    split_limit: int,
    processor: CustomWav2Vec2Processor,
    syncer: TarS3Syncer,
    split: str,
    sp_vocab_size: int,
    sp_bpe_dropout: float,
    sync_on_start: bool,
) -> ResizedDataset:
    """Create a common voice split with caching."""
    loader = LazyLoader(
        processor=processor,
        target_lang=target_lang,
        sample_rate=sample_rate,
        split=split,
        data_seed=data_seed,
        sp_dir=f"./.app_cache/sp/common_voice_{target_lang}/{split}_set/{sp_vocab_size}",
        sp_vocab_size=sp_vocab_size,
        sp_bpe_dropout=sp_bpe_dropout,
    )

    common_voice_split = LazyDataset(
        loader.load_common_voice_for_wav2vec2,
        raw_split_size,
    )
    meta_common_voice_split = LazyDataset(
        loader.load_meta_common_voice_for_wav2vec2,
        raw_split_size,
    )
    common_voice_split = ResizedDataset(common_voice_split, split_size)
    meta_common_voice_split = ResizedDataset(meta_common_voice_split, split_size)
    cache_path = f"./.app_cache/{data_seed}/{split}_set/"
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    cache_bucket = f"{target_hf_repo}-cache-{data_seed}-{split}-set"
    common_voice_split = prepare_cached_dataset(
        processor.convert_tokenizer_to_bpe(),
        common_voice_split,
        meta_common_voice_split,
        sample_rate,
        cache_path,
        cache_bucket,
        syncer,
        sync_on_start=sync_on_start,
    )

    common_voice_split = ResizedDataset(common_voice_split, split_limit)
    item = common_voice_split[0]
    print("=== Index 0 ===")
    print("Keys:", item.keys())
    print("Sentence:", item["sentence"])
    print("Labels:\t", item["labels"])
    for i in range(10):
        item = common_voice_split[0]
        print(f"Labels [{i}]:\t", item["labels"])
    return common_voice_split
