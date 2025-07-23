"""A Collator for training with CTC."""

from dataclasses import dataclass

import torch

from ..custom_wav2vec2.processor import (
    CustomWav2Vec2BertProcessor,
    CustomWav2Vec2Processor,
)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: CustomWav2Vec2Processor | CustomWav2Vec2BertProcessor
    features_name: str
    padding: bool | str = True

    def __call__(
        self, features: list[dict[str, list[int] | torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collates the features."""
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {self.features_name: feature[self.features_name]} for feature in features
        ]
        label_features = [feature["labels"] for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        batch["flat_labels"] = torch.cat(
            [torch.tensor(labels["input_ids"]) for labels in label_features]
        )  # type: ignore

        return batch
