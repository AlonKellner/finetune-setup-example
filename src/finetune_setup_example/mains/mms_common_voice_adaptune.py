"""Code for "adaptuning" mms checkpoints with common voice."""

import torch

from ..custom_datasets.resized import ResizedDataset
from ..custom_hf.trainer import train
from ..custom_hf.training_args import create_training_arguments
from ..init_utils import init_training
from ..s3_utils import create_s3_client
from ..specific_datasets.common_voice import (
    create_cached_common_voice_split,
)
from ..specific_wav2vec2.hf_utils import demo_trained_model, hf_push_adapter
from ..specific_wav2vec2.model import load_wav2vec2_for_adaptuning
from ..specific_wav2vec2.processor import create_wav2vec2_processor
from ..specific_wav2vec2.trainer import create_trainer
from ..tar_s3 import TarS3Syncer


def main(
    seed: int = 44,
    data_seed: int = 44,
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
    eval_limit: int = 10_000,
    num_train_epochs: int | float | None = 1,
    num_training_steps: int | None = None,
    effective_batch_size: int = 128,
    per_device_train_batch_size: int = 128,
    per_device_eval_batch_size: int = 128,
    per_device_train_batch_total_seconds: float = 90.0,
    per_device_eval_batch_total_seconds: float = 90.0,
    num_devices: int = 1,
    pretrained_learning_rate: float = 1e-5,
    pretrained_min_lr_ratio: float = 0.0,
    pretrained_wait_ratio: float = 0.1,
    pretrained_warmup_ratio: float = 0.1,
    pretrained_stable_ratio: float = 0.3,
    pretrained_decay_ratio: float = 0.5,
    adapter_learning_rate: float = 1e-3,
    adapter_min_lr_ratio: float = 1e-3,
    adapter_wait_ratio: float = 0.0,
    adapter_warmup_ratio: float = 0.05,
    adapter_stable_ratio: float = 0.45,
    adapter_decay_ratio: float = 0.5,
    mega_batch_mult: int = 100,
    dataloader_num_workers: int = 0,
    fp16: bool = True,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 10,
    weight_decay: float = 0.01,
    torch_compile: bool = False,
    attn_implementation: str = "flash_attention_2",
    hidden_dropout: float = 0.0,
    activation_dropout: float = 0.0,
    attention_dropout: float = 0.0,
    feat_proj_dropout: float = 0.0,
    feat_quantizer_dropout: float = 0.0,
    final_dropout: float = 0.0,
    layerdrop: float = 0.0,
    padding_side: str = "random",
    should_train: bool = True,
    should_push: bool = False,
    should_demo: bool = False,
    eval_on_start: bool = False,
    sync_on_start: bool = False,
    should_freeze_base_model: bool = False,
    should_freeze_feature_encoder: bool = True,
    sp_vocab_size: int = 64,
    sp_bpe_dropout: float = 0.1,
    job_path: str | None = None,
    hp_set: dict | None = None,
    **kwargs: str,
) -> None:
    """Training a model."""
    accelerator_available = torch.accelerator.is_available()
    if not accelerator_available:
        print("WARNING: Accelerator not available! using SDPA (not flash-attn)")
        attn_implementation = "sdpa"

    if job_path is not None:
        print(f"Job path: {job_path}")

    kwargs = {k: v for k, v in kwargs.items() if "job" not in k}
    if len(kwargs) > 0:
        print("WARNING: Got unknown kwargs.\n", kwargs)

    init_training(
        seed=seed,
        base_hf_repo=base_hf_repo,
        tokenizer_hf_repo=tokenizer_hf_repo,
        target_hf_repo=target_hf_repo,
    )

    s3_client, s3_client_v2 = create_s3_client()

    syncer = TarS3Syncer(
        s3_client=s3_client,
        s3_client_v2=s3_client_v2,
    )

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
        pretrained_learning_rate=pretrained_learning_rate,
        pretrained_min_lr_ratio=pretrained_min_lr_ratio,
        pretrained_wait_ratio=pretrained_wait_ratio,
        pretrained_warmup_ratio=pretrained_warmup_ratio,
        pretrained_stable_ratio=pretrained_stable_ratio,
        pretrained_decay_ratio=pretrained_decay_ratio,
        adapter_learning_rate=adapter_learning_rate,
        adapter_min_lr_ratio=adapter_min_lr_ratio,
        adapter_wait_ratio=adapter_wait_ratio,
        adapter_warmup_ratio=adapter_warmup_ratio,
        adapter_stable_ratio=adapter_stable_ratio,
        adapter_decay_ratio=adapter_decay_ratio,
        mega_batch_mult=mega_batch_mult,
        dataloader_num_workers=dataloader_num_workers,
        fp16=fp16,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        weight_decay=weight_decay,
        torch_compile=torch_compile,
        sample_rate=sample_rate,
        eval_on_start=eval_on_start,
        hp_set=hp_set,
    )

    train_processor, _ = create_wav2vec2_processor(
        split="train",
        target_lang=target_lang,
        sample_rate=sample_rate,
        tokenizer_hf_repo=tokenizer_hf_repo,
        target_hf_repo=target_hf_repo,
        max_batch_length=training_args.per_device_train_batch_total_length,
        padding_side=padding_side,
        syncer=syncer,
        sp_vocab_size=sp_vocab_size,
        sp_bpe_dropout=sp_bpe_dropout,
    )

    eval_processor, _ = create_wav2vec2_processor(
        split="test",
        target_lang=target_lang,
        sample_rate=sample_rate,
        tokenizer_hf_repo=tokenizer_hf_repo,
        target_hf_repo=target_hf_repo,
        max_batch_length=training_args.per_device_eval_batch_total_length,
        padding_side=padding_side,
        syncer=syncer,
        sp_vocab_size=sp_vocab_size,
        sp_bpe_dropout=sp_bpe_dropout,
    )

    if train_processor.can_create_bpe_tokenizer():
        eval_processor.set_bpe_tokenizer(train_processor.convert_tokenizer_to_bpe())

    common_voice_train = create_cached_common_voice_split(
        data_seed=data_seed,
        target_lang=target_lang,
        sample_rate=sample_rate,
        target_hf_repo=target_hf_repo,
        raw_split_size=raw_train_size,
        split_size=train_size,
        split_limit=train_limit,
        processor=train_processor,
        syncer=syncer,
        sync_on_start=sync_on_start,
        split="train",
    )

    if train_processor.can_create_bpe_tokenizer():
        eval_processor.set_bpe_tokenizer(train_processor.convert_tokenizer_to_bpe())

    common_voice_eval = create_cached_common_voice_split(
        data_seed=data_seed,
        target_lang=target_lang,
        sample_rate=sample_rate,
        target_hf_repo=target_hf_repo,
        raw_split_size=raw_eval_size,
        split_size=eval_size,
        split_limit=eval_limit,
        processor=eval_processor,
        syncer=syncer,
        sync_on_start=sync_on_start,
        split="test",
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
        should_freeze_base_model=should_freeze_base_model,
        should_freeze_feature_encoder=should_freeze_feature_encoder,
    )

    trainer = create_trainer(
        model=model,
        training_args=training_args,
        common_voice_eval=common_voice_eval,
        common_voice_train=common_voice_train,
        train_processor=train_processor,
        eval_processor=eval_processor,
    )

    if not accelerator_available:
        print(
            "WARNING: Torch accelerator is not available. Training will not be performed."
        )
        train_dataloader = trainer.get_train_dataloader()
        for item in train_dataloader:
            print([k for k in item])
            break

        metrics = trainer.evaluate(ResizedDataset(common_voice_eval, size=10))
        print(metrics)

    if should_train and accelerator_available:
        train(trainer)

    if should_push:
        hf_push_adapter(target_lang, model, training_args.output_dir, trainer)

    if should_demo:
        demo_trained_model(target_lang, sample_rate, target_hf_repo, hf_user)

    print("FINISHED!")
