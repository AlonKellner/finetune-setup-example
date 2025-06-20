"""Finetune setup utilities for initializing training and logging in to Hugging Face."""

import os

import huggingface_hub as hf_hub
from transformers import set_seed

from .printing_utils import print_basics


def init_training(
    seed: int, base_hf_repo: str, tokenizer_hf_repo: str, target_hf_repo: str
) -> None:
    """Initialize training setup and login to Hugging Face."""
    if (job_id := os.getenv("JOB_FULL_ID")) is not None:
        print(f"Running job with ID: {job_id}")

    hf_login()

    print_basics(base_hf_repo, tokenizer_hf_repo, target_hf_repo)

    set_seed(seed)


def hf_login() -> None:
    """Login to Hugging Face."""
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("Env var `HF_TOKEN` not provided.")
    hf_hub.login(
        token=token
    )  # a warning is raised about this, but the token must be set explicitly to avoid an error
    print("Logged in to Hugging Face successfully.")
