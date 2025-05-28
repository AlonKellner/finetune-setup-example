"""Wav2Vec2 HF utils."""

import os

import torch
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from safetensors.torch import save_file as safe_save_file
from transformers import Trainer, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

from ..custom_wav2vec2.wav2vec2_for_ctc import CustomWav2Vec2ForCTC
from .processor import HasCustomFields


def demo_trained_model(
    target_lang: str, sample_rate: int, target_hf_repo: str, hf_user: str
) -> None:
    """Demo trained model."""
    model_id = f"{hf_user}/{target_hf_repo}"

    model = CustomWav2Vec2ForCTC.from_pretrained(
        model_id, target_lang=target_lang, ignore_mismatched_sizes=True
    ).to(
        "cuda"  # type: ignore
    )
    _processor = Wav2Vec2Processor.from_pretrained(model_id)
    assert isinstance(_processor, HasCustomFields) and isinstance(
        _processor, Wav2Vec2Processor
    )
    processor = _processor

    processor.tokenizer.set_target_lang(target_lang)  # type: ignore

    common_voice_eval_tr: HFDataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "tr",
        data_dir="./cv-corpus-17.0-2024-03-20",
        split="test",
        token=True,
        trust_remote_code=True,
    )  # type: ignore
    common_voice_eval_tr = common_voice_eval_tr.cast_column(
        "audio", Audio(sampling_rate=sample_rate)
    )

    input_dict = processor(
        common_voice_eval_tr[0]["audio"]["array"],
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    logits = model(input_dict.input_values.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]

    print("Prediction:")
    print(processor.decode(pred_ids))

    print("\nReference:")
    print(common_voice_eval_tr[0]["sentence"].lower())


def hf_push_adapter(
    target_lang: str,
    model: CustomWav2Vec2ForCTC,
    output_dir: str | None,
    trainer: Trainer,
) -> None:
    """Push adapter to hf."""
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
    assert output_dir is not None
    adapter_file = os.path.join(output_dir, adapter_file)

    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

    trainer.push_to_hub()
