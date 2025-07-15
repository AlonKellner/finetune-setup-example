"""Utilities for wav2vec2 processors."""

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)

from ..custom_wav2vec2.feature_extractor import CustomWav2Vec2FeatureExtractor
from ..custom_wav2vec2.processor import CustomWav2Vec2Processor


@runtime_checkable
class HasCustomFields(Protocol):
    """Just for type checking."""

    tokenizer: Wav2Vec2CTCTokenizer
    feature_extractor: Wav2Vec2FeatureExtractor


def create_wav2vec2_processor(
    split: str,
    target_lang: str,
    sample_rate: int,
    tokenizer_hf_repo: str,
    target_hf_repo: str,
    sp_bpe_dropout: float,
    max_batch_length: int | None = None,
    padding_side: str = "random",
) -> tuple[CustomWav2Vec2Processor, CustomWav2Vec2FeatureExtractor]:
    """Create a wav2vec2 processor, ready for training a specific language."""
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_hf_repo)
    vocab_dict = tokenizer.vocab

    new_vocab_dict = {target_lang: vocab_dict}

    vocab_file = Path(".output/vocab.json")
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, "w") as f:
        json.dump(new_vocab_dict, f)

    tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        vocab_file.parent,
        unk_token=tokenizer.unk_token,
        pad_token=tokenizer.pad_token,
        word_delimiter_token=tokenizer.word_delimiter_token,
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token,
        target_lang=target_lang,
    )

    tokenizer.push_to_hub(target_hf_repo)  # type: ignore

    feature_extractor = CustomWav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
        max_batch_length=max_batch_length,
        padding_side=padding_side,
    )

    processor: CustomWav2Vec2Processor = CustomWav2Vec2Processor(
        sp_dir=f"./.app_cache/sp/common_voice_{target_lang}/{split}_set/spm",
        sp_bpe_dropout=sp_bpe_dropout,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    assert isinstance(processor, HasCustomFields) and isinstance(
        processor, CustomWav2Vec2Processor
    )
    return processor, feature_extractor
