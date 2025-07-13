"""A custom sentencepiece tokenizer for Wav2Vec2."""

import os
from shutil import copyfile

import sentencepiece as spm
from transformers import Wav2Vec2CTCTokenizer


class BpeWav2Vec2CTCTokenizer(Wav2Vec2CTCTokenizer):
    """Custom BPE tokenizer for Wav2Vec2 using SentencePiece."""

    def __init__(
        self,
        sp_model_path: str,
        **kwargs: object,
    ) -> None:
        self.sp_model_path = sp_model_path
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(sp_model_path)

        vocab = {
            self.sp_model.id_to_piece(i): i
            for i in range(self.sp_model.get_piece_size())
        }
        self.vocab_dict = vocab

        super().__init__(
            vocab_file=None,
            unk_token=self.sp_model.id_to_piece(self.sp_model.unk_id()),
            pad_token="<pad>",  # noqa: S106
            bos_token="<s>",  # noqa: S106
            eos_token="</s>",  # noqa: S106
            **kwargs,
        )

        self.vocab_dict["<pad>"] = len(self.vocab_dict)
        self.vocab_dict["<s>"] = len(self.vocab_dict)
        self.vocab_dict["</s>"] = len(self.vocab_dict)

        self._additional_special_tokens = []

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize input text using the SentencePiece model."""
        return self.sp_model.encode(text, out_type=str)

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
