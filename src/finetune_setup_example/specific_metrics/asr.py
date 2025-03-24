"""Wav2Vec2 ASR metrics."""

from typing import Any

import numpy as np
from evaluate import load
from transformers import (
    EvalPrediction,
    Wav2Vec2Processor,
)

from ..specific_wav2vec2.processor import HasCustomFields

Metrics = dict[str, Any]


class Wav2Vec2ASR:
    """A wrapper for ASR metric with a wav2vec2 processor."""

    def __init__(self, processor: Wav2Vec2Processor) -> None:
        self.processor = processor
        self.wer_metric = load("wer")
        self.cer_metric = load("cer")

    def compute_metrics(self, pred: EvalPrediction) -> Metrics:
        """Compute metrics."""
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        assert isinstance(self.processor, HasCustomFields)
        if isinstance(pred.label_ids, tuple):
            pred.label_ids = pred.label_ids[0]
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id  # type: ignore

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        for i in range(100):
            print(f'({i}) pred: "{pred_str[i]}"')
            print(f'({i}) targ: "{label_str[i]}"')
        return {"wer": wer, "cer": cer}
