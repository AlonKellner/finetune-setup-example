"""A custom sentencepiece tokenizer for Wav2Vec2."""

import json
import os
import tempfile
from shutil import copyfile
from typing import Any

import sentencepiece as spm
from transformers import Wav2Vec2CTCTokenizer


class BpeWav2Vec2CTCTokenizer(Wav2Vec2CTCTokenizer):
    """Custom BPE tokenizer for Wav2Vec2 using SentencePiece."""

    def __init__(
        self,
        sp_model_path: str,
        sp_bpe_dropout: float,
        **kwargs: object,
    ) -> None:
        self.sp_model_path = sp_model_path
        self.sp_bpe_dropout = sp_bpe_dropout
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(sp_model_path)

        vocab = {
            self.sp_model.id_to_piece(i): i
            for i in range(self.sp_model.get_piece_size())
        }
        self.vocab_dict = vocab
        if (target_lang := kwargs.get("target_lang")) is not None:
            self.target_lang = target_lang
            vocab = {target_lang: vocab}
        else:
            self.target_lang = None

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            f.write(json.dumps(vocab, indent=2))
            f.seek(0)
            super().__init__(
                vocab_file=f.name,
                unk_token=self.sp_model.id_to_piece(self.sp_model.unk_id()),
                pad_token=self.sp_model.id_to_piece(self.sp_model.pad_id()),
                bos_token=self.sp_model.id_to_piece(self.sp_model.bos_id()),
                eos_token=self.sp_model.id_to_piece(self.sp_model.eos_id()),
                **kwargs,
            )

        self._additional_special_tokens = []

    def tokenize(self, text: str, **kwargs: Any) -> list[str]:
        """Tokenize input text using the SentencePiece model."""
        kwargs.pop("split_special_tokens", self.split_special_tokens)

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        return self.sp_model.encode(
            text,
            out_type=str,
            enable_sampling=True,
            alpha=self.sp_bpe_dropout,
            nbest_size=-1,
        )

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its corresponding id."""
        return self.vocab_dict.get(token, self.vocab_dict.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an id to its corresponding token."""
        for k, v in self.vocab_dict.items():
            if v == index:
                return k
        return self.unk_token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert a list of tokens back to a string."""
        return self.sp_model.decode_pieces(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        """Save the SentencePiece model to the specified directory."""
        output_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "sp.model",
        )
        copyfile(self.sp_model_path, output_vocab_file)
        return (output_vocab_file,)
