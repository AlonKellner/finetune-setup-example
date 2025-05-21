"""Will be an entrypoint for job training."""

from datetime import datetime, timedelta

from finetune_setup_example.local_job_utils import run_local_job


def train_main(wait_seconds: int, prefix: str) -> None:
    """Train a model with a given wait time and print a prefixed counter."""
    start = datetime.now()
    counter = 0
    while datetime.now() - start < timedelta(seconds=wait_seconds):
        counter += 1
    print(f"{prefix}: {counter}")


if __name__ == "__main__":
    run_local_job(train_main)
