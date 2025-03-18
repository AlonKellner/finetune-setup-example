"""Common voice loading utilities."""

import random
import re
from typing import Any

import uroman
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from transformers import Wav2Vec2Processor

Batch = dict[str, Any]


def load_common_voice_for_wav2vec2(
    processor: Wav2Vec2Processor,
    target_lang: str,
    sample_rate: int,
    split: str,
    data_seed: int,
) -> HFDataset:
    """Load a split of common voice 17, adapted for wav2vec2."""
    common_voice_split = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "tr",
        split=split,
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

    chars_to_remove_regex = r"[`,?\.!\-;:\"“%‘”�()…’]"  # noqa: RUF001
    ur = uroman.Uroman()

    def uromanize(batch: Batch) -> Batch:
        """Uromanize text."""
        clean_string = re.sub(chars_to_remove_regex, "", batch["sentence"]).lower()
        batch["sentence"] = ur.romanize_string(clean_string, lcode=target_lang)
        return batch

    common_voice_split = common_voice_split.map(uromanize)

    common_voice_split = common_voice_split.cast_column(
        "audio", Audio(sampling_rate=sample_rate)
    )

    rand_int = random.randint(0, len(common_voice_split) - 1)

    print("Target text:", common_voice_split[rand_int]["sentence"])
    print("Input array shape:", common_voice_split[rand_int]["audio"]["array"].shape)
    print("Sampling rate:", common_voice_split[rand_int]["audio"]["sampling_rate"])

    def prepare_dataset(batch: Batch) -> Batch:
        """Prepare dataset."""
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["length"] = len(batch["input_values"])

        batch["labels"] = processor(text=batch["sentence"]).input_ids  # type: ignore
        return batch

    common_voice_split = common_voice_split.map(
        prepare_dataset, remove_columns=common_voice_split.column_names
    )
    common_voice_split = common_voice_split.shuffle(seed=data_seed)

    return common_voice_split
