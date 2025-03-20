"""A mockup of multi-HP jobs."""

import threading
import time
from pathlib import Path

from finetune_setup_example.experiment_utils import get_exp_ids
from finetune_setup_example.hp_set_handling import (
    load_hp_set,
    prepare_hp_sets,
)


def train_main(hp_path: Path, job_id: str, ids: dict[str, str]) -> None:
    """Train a model. A mock train main."""
    hp_set = load_hp_set(hp_path)
    print(job_id, "\t", ids, "\t", hp_path, "\t", hp_set)


def run_job(hp_path: Path, ids: dict[str, str]) -> str:
    """Mock job running logic."""
    job_id = "-".join(ids.values())
    thread = threading.Thread(
        target=train_main,
        args=(
            hp_path,
            job_id,
            ids,
        ),
    )
    thread.start()
    return job_id


def run_jobs(hp_paths: list[tuple[str, Path]], ids: dict[str, str]) -> None:
    """Run multiple jobs for an experiment."""
    job_ids = [run_job(hp_path, dict(**ids, i=i)) for (i, hp_path) in hp_paths]
    print(job_ids)
    time.sleep(1)


def hp_main() -> None:
    """Create hyper-parameter sets and run jobs for them."""
    default_hp_set = dict(a=1, b=2, c=3)
    hp_sets = [dict(b=3, c=c, z=0, x=x) for c in [-1, -2] for x in [0.4, 0.5]]
    hp_paths = prepare_hp_sets(default_hp_set, hp_sets, Path("hp_sets"))

    ids = get_exp_ids()

    run_jobs(hp_paths, ids)


if __name__ == "__main__":
    hp_main()
