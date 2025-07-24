"""Wav2Vec2 HF utils."""

from typing import Literal

import torch
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from safetensors.torch import save_file as safe_save_file
from transformers import Trainer, Wav2Vec2Processor

from ..custom_wav2vec2.model_for_ctc import (
    CustomWav2Vec2BertForCTC,
    CustomWav2Vec2ForCTC,
)
from .processor import HasCustomFields


def demo_trained_model(
    target_lang: str,
    sample_rate: int,
    target_hf_repo: str,
    hf_user: str,
    architecture: Literal["wav2vec2", "w2v-bert2"],
) -> None:
    """Demo trained model."""
    model_id = f"{hf_user}/{target_hf_repo}"

    if architecture == "wav2vec2":
        ctc_model_type = CustomWav2Vec2ForCTC
        features_name = "input_values"
    elif architecture == "w2v-bert2":
        ctc_model_type = CustomWav2Vec2BertForCTC
        features_name = "input_features"
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = ctc_model_type.from_pretrained(
        model_id, target_lang=target_lang, ignore_mismatched_sizes=True
    ).to(
        "cuda"  # type: ignore
    )
    _processor = Wav2Vec2Processor.from_pretrained(model_id)
    assert isinstance(_processor, HasCustomFields) and isinstance(
        _processor, Wav2Vec2Processor
    )
    processor = _processor

    processor.tokenizer.set_target_lang(target_lang)

    common_voice_eval_tr: HFDataset = load_dataset(  # type: ignore
        "mozilla-foundation/common_voice_17_0",
        "tr",
        data_dir="./cv-corpus-17.0-2024-03-20",
        split="test",
        token=True,
        trust_remote_code=True,
    )
    common_voice_eval_tr = common_voice_eval_tr.cast_column(
        "audio", Audio(sampling_rate=sample_rate)
    )

    input_dict = processor(
        common_voice_eval_tr[0]["audio"]["array"],
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    logits = model(input_dict[features_name].to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]

    print("Prediction:")
    print(processor.decode(pred_ids))

    print("\nReference:")
    print(common_voice_eval_tr[0]["sentence"].lower())


def hf_push_adapter(
    target_lang: str,
    model: CustomWav2Vec2ForCTC | CustomWav2Vec2BertForCTC,
    output_dir: str | None,
    trainer: Trainer,
) -> None:
    """Push adapter to hf."""
    assert output_dir is not None

    safe_save_file(
        model.state_dict(), f"{target_lang}.safetensors", metadata={"format": "pt"}
    )

    trainer.push_to_hub()  # type: ignore
