"""Utilities for creating wav2vec2 trainers."""

from transformers import Wav2Vec2Processor

from finetune_setup_example.custom_datasets.resized import ResizedDataset
from finetune_setup_example.custom_datasets.tar_s3 import TarS3Dataset
from finetune_setup_example.custom_hf.trainer import CustomTrainer
from finetune_setup_example.custom_hf.training_args import CustomTrainingArguments
from finetune_setup_example.custom_wav2vec2.ctc_collator import (
    DataCollatorCTCWithPadding,
)
from finetune_setup_example.custom_wav2vec2.wav2vec2_for_ctc import CustomWav2Vec2ForCTC
from finetune_setup_example.specific_metrics.asr import Wav2Vec2ASR
from finetune_setup_example.specific_wav2vec2.processor import HasCustomFields


def create_trainer(
    model: CustomWav2Vec2ForCTC,
    training_args: CustomTrainingArguments,
    common_voice_test: TarS3Dataset | ResizedDataset,
    common_voice_train: TarS3Dataset | ResizedDataset,
    processor: Wav2Vec2Processor,
) -> CustomTrainer:
    """Create a wav2vec2 common voice trainer."""
    assert isinstance(processor, HasCustomFields)
    test_indices_order = list(range(len(common_voice_test)))
    train_indices_order = list(range(len(common_voice_train)))
    wav2vec2_asr = Wav2Vec2ASR(processor=processor)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    trainer = CustomTrainer(
        model=model,  # type: ignore
        data_collator=data_collator,
        args=training_args,
        compute_metrics=wav2vec2_asr.compute_metrics,
        eval_dataset=common_voice_test,
        train_dataset=common_voice_train,
        processing_class=processor.feature_extractor,
        test_indices_order=test_indices_order,
        train_indices_order=train_indices_order,
        test_grouped_indices=common_voice_test.sync_safe_groups,
        train_grouped_indices=common_voice_train.sync_safe_groups,
    )
    return trainer
