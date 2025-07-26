"""Utilities for printing useful information."""

import torchaudio as ta


def print_basics(
    base_hf_repo: str, tokenizer_hf_repo: str, target_hf_repo: str
) -> None:
    """Print basic stuff for visibility."""
    print(f"torchaudio backends: {ta.list_audio_backends()}")

    print(f"base_hf_repo: {base_hf_repo}")
    print(f"tokenizer_hf_repo: {tokenizer_hf_repo}")
    print(f"target_hf_repo: {target_hf_repo}")
