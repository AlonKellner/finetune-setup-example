"""Custom Wav2Vec2 Processor."""

from pathlib import Path
from typing import Any

import sentencepiece as spm
from transformers import Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

from ..tar_s3 import TarS3Syncer
from .tokenizer import BpeWav2Vec2CTCTokenizer


class CustomProcessorMixin:
    """Custom Wav2Vec2Processor."""

    def __init__(
        self,
        *args: Any,
        sp_dir: str,
        sp_bpe_dropout: float,
        sp_vocab_size: int,
        syncer: TarS3Syncer,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sp_dir = sp_dir
        self.sp_bpe_dropout = sp_bpe_dropout
        self.sp_vocab_size = sp_vocab_size
        self.syncer = syncer
        self.is_converted = False
        self.sp_model_path = Path(f"{self.sp_dir}/spm.model")
        self.sp_files = [
            self.sp_model_path,
            Path(f"{self.sp_dir}/spm.vocab"),
            Path(f"{self.sp_dir}/raw.txt"),
        ]
        self.bucket = "finetune-setup-example-sp"
        if not self.syncer._bucket_exists(self.bucket):
            self.syncer._create_bucket(self.bucket)

    tokenizer: Wav2Vec2CTCTokenizer

    def can_create_bpe_tokenizer(self) -> bool:
        """Check if the tokenizer can be converted to BPE."""
        self.syncer.sync_multiple_files(
            self.sp_files, self.sp_dir, self.bucket, Path(self.sp_dir)
        )
        return self.sp_model_path.exists() and not self.is_converted

    def set_bpe_tokenizer(self, tokenizer: BpeWav2Vec2CTCTokenizer) -> None:
        """Set the tokenizer."""
        self.tokenizer = tokenizer
        self.is_converted = True

    def convert_tokenizer_to_bpe(self) -> BpeWav2Vec2CTCTokenizer:
        """Create a tokenizer."""
        if not self.is_converted:
            self.tokenizer = BpeWav2Vec2CTCTokenizer(
                sp_model_path=str(self.sp_model_path),
                sp_bpe_dropout=self.sp_bpe_dropout,
                word_delimiter_token=self.tokenizer.word_delimiter_token,
                target_lang=self.tokenizer.target_lang,
            )
            self.is_converted = True
        return self.tokenizer

    def train_bpe_tokenizer(self, data: list[str]) -> None:
        """Train BPE tokenizer."""
        sp_dir = self.sp_dir
        train_text = "\n".join(data)
        train_text_path = f"{sp_dir}/raw.txt"
        Path(sp_dir).mkdir(parents=True, exist_ok=True)
        with open(train_text_path, "w") as f:
            f.write(train_text)
        spm.SentencePieceTrainer.Train(
            input=train_text_path,
            model_prefix=f"{sp_dir}/spm",
            character_coverage=1.0,
            vocab_size=self.sp_vocab_size,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )


class CustomWav2Vec2Processor(CustomProcessorMixin, Wav2Vec2Processor):
    """Wav2Vec2 processor with custom mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class CustomWav2Vec2BertProcessor(CustomProcessorMixin, Wav2Vec2BertProcessor):
    """Wav2Vec2-Bert processor with custom mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
