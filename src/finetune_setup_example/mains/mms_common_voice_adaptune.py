"""Code for "adaptuning" mms checkpoints with common voice."""

import os

import comet_ml  # type: ignore  # noqa: F401
import huggingface_hub as hf_hub
from transformers import set_seed

from ..custom_hf.trainer import train
from ..custom_hf.training_args import create_training_arguments
from ..printing_utils import print_basics
from ..s3_utils import create_s3_client
from ..specific_datasets.common_voice import (
    create_cached_common_voice_split,
)
from ..specific_wav2vec2.hf_utils import demo_trained_model, hf_push_adapter
from ..specific_wav2vec2.model import load_wav2vec2_for_adaptuning
from ..specific_wav2vec2.processor import create_wav2vec2_processor
from ..specific_wav2vec2.trainer import create_trainer


def main(
    seed: int = 42,
    data_seed: int = 42,
    target_lang: str = "tur",
    sample_rate: int = 16_000,
    base_hf_repo: str = "mms-meta/mms-zeroshot-300m",
    tokenizer_hf_repo: str = "mms-meta/mms-zeroshot-300m",
    target_hf_repo: str = "mms-300m-turkish",
    hf_user: str = "Kellner",
    raw_train_size: int = 35147,
    raw_eval_size: int = 11290,
    eval_size: int = 20_000,
    train_size: int = 100_000,
    train_limit: int = 100_000,
    eval_limit: int = 1_000,
    num_train_epochs: int | float | None = 1,
    num_training_steps: int | None = None,
    effective_batch_size: int = 16,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 8,
    per_device_train_batch_total_seconds: float = 90.0,
    per_device_eval_batch_total_seconds: float = 90.0,
    num_devices: int = 1,
    warmup_ratio: float = 0.0,
    decay_ratio: float = 1.0,
    learning_rate: float = 1e-2,
    mega_batch_mult: int = 100,
    dataloader_num_workers: int = 4,
    fp16: bool = False,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 10,
    weight_decay: float = 0.01,
    torch_compile: bool = False,
    attn_implementation: str = "sdpa",
    hidden_dropout: float = 0.0,
    activation_dropout: float = 0.0,
    attention_dropout: float = 0.0,
    feat_proj_dropout: float = 0.0,
    feat_quantizer_dropout: float = 0.0,
    final_dropout: float = 0.0,
    layerdrop: float = 0.0,
    padding_side: str = "random",
    should_push: bool = False,
    should_demo: bool = False,
    eval_on_start: bool = True,
    steps_per_epoch: int | None = 1050,
) -> None:
    """Training a model."""
    hf_hub.login(os.getenv("HF_TOKEN"), new_session=False)

    print_basics(base_hf_repo, tokenizer_hf_repo, target_hf_repo)

    set_seed(seed)

    training_args = create_training_arguments(
        seed=seed,
        data_seed=data_seed,
        target_hf_repo=target_hf_repo,
        num_train_epochs=num_train_epochs,
        num_training_steps=num_training_steps,
        effective_batch_size=effective_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        per_device_train_batch_total_seconds=per_device_train_batch_total_seconds,
        per_device_eval_batch_total_seconds=per_device_eval_batch_total_seconds,
        num_devices=num_devices,
        warmup_ratio=warmup_ratio,
        decay_ratio=decay_ratio,
        learning_rate=learning_rate,
        mega_batch_mult=mega_batch_mult,
        dataloader_num_workers=dataloader_num_workers,
        fp16=fp16,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        weight_decay=weight_decay,
        torch_compile=torch_compile,
        train_size=train_limit,
        sample_rate=sample_rate,
        eval_on_start=eval_on_start,
        steps_per_epoch=steps_per_epoch,
    )

    train_processor, _ = create_wav2vec2_processor(
        target_lang=target_lang,
        sample_rate=sample_rate,
        tokenizer_hf_repo=tokenizer_hf_repo,
        target_hf_repo=target_hf_repo,
        max_batch_length=training_args.per_device_train_batch_total_length,
        padding_side=padding_side,
    )

    eval_processor, _ = create_wav2vec2_processor(
        target_lang=target_lang,
        sample_rate=sample_rate,
        tokenizer_hf_repo=tokenizer_hf_repo,
        target_hf_repo=target_hf_repo,
        max_batch_length=training_args.per_device_eval_batch_total_length,
        padding_side=padding_side,
    )

    s3_client, s3_client_v2 = create_s3_client()

    common_voice_train = create_cached_common_voice_split(
        data_seed,
        target_lang,
        sample_rate,
        target_hf_repo,
        raw_train_size,
        train_size,
        train_limit,
        train_processor,
        s3_client,
        s3_client_v2,
        "train",
    )
    common_voice_eval = create_cached_common_voice_split(
        data_seed,
        target_lang,
        sample_rate,
        target_hf_repo,
        raw_eval_size,
        eval_size,
        eval_limit,
        eval_processor,
        s3_client,
        s3_client_v2,
        "test",
    )

    model = load_wav2vec2_for_adaptuning(
        base_hf_repo=base_hf_repo,
        processor=train_processor,
        attn_implementation=attn_implementation,
        hidden_dropout=hidden_dropout,
        activation_dropout=activation_dropout,
        attention_dropout=attention_dropout,
        feat_proj_dropout=feat_proj_dropout,
        feat_quantizer_dropout=feat_quantizer_dropout,
        final_dropout=final_dropout,
        layerdrop=layerdrop,
    )

    trainer = create_trainer(
        model=model,
        training_args=training_args,
        common_voice_eval=common_voice_eval,
        common_voice_train=common_voice_train,
        train_processor=train_processor,
        eval_processor=eval_processor,
    )
    train(trainer)

    if should_push:
        hf_push_adapter(target_lang, model, training_args.output_dir, trainer)

    if should_demo:
        demo_trained_model(target_lang, sample_rate, target_hf_repo, hf_user)
