"""Custom Wav2Vec2 Processor."""

from pathlib import Path
from typing import Any

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)

from ..specific_wav2vec2.tokenizer import BpeWav2Vec2CTCTokenizer


class CustomWav2Vec2Processor(Wav2Vec2Processor):
    """Custom Wav2Vec2Processor."""

    def __init__(
        self, *args: Any, sp_dir: str, sp_bpe_dropout: float, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sp_dir = sp_dir
        self.sp_bpe_dropout = sp_bpe_dropout
        self.is_converted = False

    tokenizer: Wav2Vec2CTCTokenizer

    def can_create_bpe_tokenizer(self) -> bool:
        """Check if the tokenizer can be converted to BPE."""
        return Path(f"{self.sp_dir}/spm.model").exists() and not self.is_converted

    def set_bpe_tokenizer(self, tokenizer: BpeWav2Vec2CTCTokenizer) -> None:
        """Set the tokenizer."""
        self.tokenizer = tokenizer
        self.is_converted = True

    def convert_tokenizer_to_bpe(self) -> BpeWav2Vec2CTCTokenizer:
        """Create a tokenizer."""
        if not self.is_converted:
            self.tokenizer = BpeWav2Vec2CTCTokenizer(
                sp_model_path=f"{self.sp_dir}/spm.model",
                sp_bpe_dropout=self.sp_bpe_dropout,
                word_delimiter_token=self.tokenizer.word_delimiter_token,
                target_lang=self.tokenizer.target_lang,
            )
            self.is_converted = True
        return self.tokenizer
