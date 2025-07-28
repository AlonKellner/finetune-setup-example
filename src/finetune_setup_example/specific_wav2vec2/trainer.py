"""Utilities for creating wav2vec2 trainers."""

from torch.utils.data import Dataset as TorchDataset

from ..custom_hf.trainer import CustomTrainer
from ..custom_hf.training_args import CustomTrainingArguments
from ..custom_wav2vec2.ctc_collator import (
    DataCollatorCTCWithPadding,
)
from ..custom_wav2vec2.model_for_ctc import (
    CustomWav2Vec2BertForCTC,
    CustomWav2Vec2ForCTC,
)
from ..custom_wav2vec2.processor import (
    CustomWav2Vec2BertProcessor,
    CustomWav2Vec2Processor,
)
from ..specific_metrics.asr import Wav2Vec2ASR
from ..specific_wav2vec2.processor import HasCustomFields


def create_trainer(
    model: CustomWav2Vec2ForCTC | CustomWav2Vec2BertForCTC,
    training_args: CustomTrainingArguments,
    common_voice_eval: TorchDataset,
    eval_grouped_indices: list[list[int]],
    common_voice_train: TorchDataset,
    train_grouped_indices: list[list[int]],
    train_processor: CustomWav2Vec2Processor | CustomWav2Vec2BertProcessor,
    eval_processor: CustomWav2Vec2Processor | CustomWav2Vec2BertProcessor,
    features_name: str,
) -> CustomTrainer:
    """Create a wav2vec2 common voice trainer."""
    assert isinstance(train_processor, HasCustomFields)
    assert isinstance(eval_processor, HasCustomFields)
    wav2vec2_asr = Wav2Vec2ASR(processor=eval_processor)
    train_data_collator = DataCollatorCTCWithPadding(
        processor=train_processor, padding=True, features_name=features_name
    )
    eval_data_collator = DataCollatorCTCWithPadding(
        processor=eval_processor, padding=True, features_name=features_name
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
        eval_grouped_indices=eval_grouped_indices,
        train_grouped_indices=train_grouped_indices,
    )
    return trainer
