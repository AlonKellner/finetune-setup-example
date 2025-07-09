"""Utilities for creating wav2vec2 trainers."""

from transformers import Wav2Vec2Processor

from ..custom_datasets.resized import ResizedDataset
from ..custom_datasets.tar_s3 import TarS3Dataset
from ..custom_hf.trainer import CustomTrainer
from ..custom_hf.training_args import CustomTrainingArguments
from ..custom_wav2vec2.ctc_collator import (
    DataCollatorCTCWithPadding,
)
from ..custom_wav2vec2.wav2vec2_for_ctc import CustomWav2Vec2ForCTC
from ..specific_metrics.asr import Wav2Vec2ASR
from ..specific_wav2vec2.processor import HasCustomFields


def create_trainer(
    model: CustomWav2Vec2ForCTC,
    training_args: CustomTrainingArguments,
    common_voice_eval: TarS3Dataset | ResizedDataset,
    common_voice_train: TarS3Dataset | ResizedDataset,
    train_processor: Wav2Vec2Processor,
    eval_processor: Wav2Vec2Processor,
) -> CustomTrainer:
    """Create a wav2vec2 common voice trainer."""
    assert isinstance(train_processor, HasCustomFields)
    assert isinstance(eval_processor, HasCustomFields)
    eval_indices_order = list(range(len(common_voice_eval)))
    train_indices_order = list(range(len(common_voice_train)))
    wav2vec2_asr = Wav2Vec2ASR(processor=eval_processor)
    train_data_collator = DataCollatorCTCWithPadding(
        processor=train_processor, padding=True
    )
    eval_data_collator = DataCollatorCTCWithPadding(
        processor=eval_processor, padding=True
    )
    trainer = CustomTrainer(
        model=model,  # type: ignore
        data_collator=train_data_collator,
        eval_data_collator=eval_data_collator,
        args=training_args,
        compute_metrics=wav2vec2_asr.compute_metrics,
        eval_dataset=common_voice_eval,
        train_dataset=common_voice_train,
        processing_class=train_processor.feature_extractor,
        eval_processing_class=eval_processor.feature_extractor,
        eval_indices_order=eval_indices_order,
        train_indices_order=train_indices_order,
        eval_grouped_indices=common_voice_eval.grouped_indices,
        train_grouped_indices=common_voice_train.grouped_indices,
    )
    trainer._get_train_sampler()
    return trainer
