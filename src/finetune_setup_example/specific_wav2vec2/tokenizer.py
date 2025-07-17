"""A custom sentencepiece tokenizer for Wav2Vec2."""

import json
import tempfile
from itertools import groupby
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

    def convert_tokens_to_string(
        self,
        tokens: list[str],
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,  # noqa: ARG002
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
    ) -> dict[str, str | float]:
        """Convert a connectionist-temporal-classification (CTC) output tokens into a single string."""
        if len(tokens) == 0:
            return {"text": "", "char_offsets": [], "word_offsets": []}
        # group same tokens into non-repeating tokens in CTC style decoding
        if group_tokens:
            chars, char_repetitions = zip(
                *(
                    (token, len(list(group_iter)))
                    for token, group_iter in groupby(tokens)
                ),
                strict=False,
            )
        else:
            chars = tokens
            char_repetitions = len(tokens) * [1]

        # filter self.pad_token which is used as CTC-blank token
        processed_chars = list(filter(lambda char: char != self.pad_token, chars))

        # replace delimiter token
        processed_chars = [
            self.replace_word_delimiter_char
            if char == self.word_delimiter_token
            else char
            for char in processed_chars
        ]

        # retrieve offsets
        char_offsets = word_offsets = None
        if output_char_offsets or output_word_offsets:
            char_offsets = self._compute_offsets(
                char_repetitions, chars, self.pad_token
            )

            if len(char_offsets) != len(processed_chars):
                raise ValueError(
                    f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                    " have to be of the same length, but are: "
                    f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                    f" {len(processed_chars)}"
                )

            # set tokens to correct processed token
            for i, char in enumerate(processed_chars):
                char_offsets[i]["char"] = char

            # retrieve word offsets from character offsets
            word_offsets = None
            if output_word_offsets:
                word_offsets = self._get_word_offsets(
                    char_offsets, self.replace_word_delimiter_char
                )

            # don't output chars if not set to True
            if not output_char_offsets:
                char_offsets = None

        # join to string
        string = self.sp_model.decode_pieces(processed_chars)
        if "⁇" in string:
            print("WARNING: unknown token")
            print("string:\t", string)
            print("processed_chars:\t", processed_chars)

        if self.do_lower_case:
            string = string.lower()

        return {
            "text": string,
            "char_offsets": char_offsets,
            "word_offsets": word_offsets,
        }
