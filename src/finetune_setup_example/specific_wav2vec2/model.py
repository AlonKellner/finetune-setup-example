"""Utilities for loading wav2vec2 models."""

from transformers import Wav2Vec2Processor

from finetune_setup_example.custom_wav2vec2.wav2vec2_for_ctc import CustomWav2Vec2ForCTC
from finetune_setup_example.specific_wav2vec2.processor import HasCustomFields


def load_wav2vec2_for_adaptuning(
    base_hf_repo: str, processor: Wav2Vec2Processor
) -> CustomWav2Vec2ForCTC:
    """Load a wav2vec2 model, ready to train an adapter alone."""
    assert isinstance(processor, HasCustomFields)
    model = CustomWav2Vec2ForCTC.from_pretrained(
        base_hf_repo,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        feat_quantizer_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        vocab_size=len(processor.tokenizer),
        adapter_attn_dim=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        # attn_implementation="flash_attention_2",
        # attn_implementation="sdpa",
    )

    model.init_adapter_layers()

    model.freeze_base_model()
    model.freeze_feature_encoder()

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    return model
