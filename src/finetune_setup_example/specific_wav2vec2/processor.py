"""Utilities for wav2vec2 processors."""

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)


@runtime_checkable
class HasCustomFields(Protocol):
    """Just for pyright type checking."""

    tokenizer: Wav2Vec2CTCTokenizer
    feature_extractor: Wav2Vec2FeatureExtractor


def create_wav2vec2_processor(
    target_lang: str, sample_rate: int, base_hf_repo: str, target_hf_repo: str
) -> Wav2Vec2Processor:
    """Create a wav2vec2 processor, ready for training a specific language."""
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(base_hf_repo)
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

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor: Wav2Vec2Processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    assert isinstance(processor, HasCustomFields) and isinstance(
        processor, Wav2Vec2Processor
    )
    return processor
