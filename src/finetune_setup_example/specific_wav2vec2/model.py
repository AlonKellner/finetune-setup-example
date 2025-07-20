"""Utilities for loading wav2vec2 models."""

from transformers import Wav2Vec2Processor

from ..custom_wav2vec2.model_for_ctc import CustomWav2Vec2ForCTC
from ..specific_wav2vec2.processor import HasCustomFields


def load_wav2vec2_for_adaptuning(
    base_hf_repo: str,
    processor: Wav2Vec2Processor,
    attn_implementation: str,
    hidden_dropout: float = 0.0,
    activation_dropout: float = 0.0,
    attention_dropout: float = 0.0,
    feat_proj_dropout: float = 0.0,
    feat_quantizer_dropout: float = 0.0,
    final_dropout: float = 0.0,
    layerdrop: float = 0.0,
    should_freeze_base_model: bool = True,
    should_freeze_feature_encoder: bool = True,
) -> CustomWav2Vec2ForCTC:
    """Load a wav2vec2 model, ready to train an adapter alone."""
    assert isinstance(processor, HasCustomFields)
    vocab_size = len(processor.tokenizer.vocab_dict)
    print(f"Vocab size: {vocab_size}")
    model = CustomWav2Vec2ForCTC.from_pretrained(
        base_hf_repo,
        hidden_dropout=hidden_dropout,
        activation_dropout=activation_dropout,
        attention_dropout=attention_dropout,
        feat_proj_dropout=feat_proj_dropout,
        feat_quantizer_dropout=feat_quantizer_dropout,
        final_dropout=final_dropout,
        layerdrop=layerdrop,
        ctc_loss_reduction="sum",
        vocab_size=vocab_size,
        adapter_attn_dim=vocab_size,
        ignore_mismatched_sizes=True,
        attn_implementation=attn_implementation,
    )

    model.init_adapter_layers()

    if should_freeze_base_model:
        model.freeze_base_model()
    if should_freeze_feature_encoder:
        model.freeze_feature_encoder()

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    return model
