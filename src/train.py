"""Will be an entrypoint for job training."""

import os
from datetime import datetime, timedelta
from pathlib import Path

from finetune_setup_example.hp_set_handling import load_hp_set


def train_main(hp_path: Path) -> None:
    """Train a model with a given hyperparameter set."""
    hp_set = load_hp_set(hp_path)  # Path object is directly used
    print("Loaded HP Set:", hp_set)

    start = datetime.now()
    counter = 0
    while datetime.now() - start < timedelta(seconds=10):
        counter += 1
    print(counter)


if __name__ == "__main__":
    hp_path = Path(os.getenv("JOB_HP_PATH", "hp_sets/0.yaml"))  # Convert to Path
    train_main(hp_path)
