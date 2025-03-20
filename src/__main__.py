"""A mockup of multi-HP jobs."""

import secrets
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from git import Repo

HPSet = dict[str, Any]


def load_hp_set(hp_path: Path) -> HPSet:
    """Load an HP set by index."""
    with open(hp_path) as f:
        hp_set = yaml.safe_load(f)
    return hp_set


def train_main(hp_path: Path, job_name: str) -> None:
    """Train a model. A mock train main."""
    id = f"{job_name}:{threading.get_ident()}"
    hp_set = load_hp_set(hp_path)
    print(id, "\t", hp_path, "\t", hp_set)


def run_job(hp_path: Path, ids: dict[str, str]) -> str:
    """Mock job running logic."""
    job_id = "-".join(ids.values())
    thread = threading.Thread(target=train_main, args=(hp_path,))
    thread.start()
    return f"{job_id}:{thread.ident}"


def to_full_hp_sets(default_hp_set: HPSet, hp_sets: list[HPSet]) -> list[HPSet]:
    """Convert partial HP sets to full with default values."""
    full_hp_sets = []
    for hp_set in hp_sets:
        full_hp_set = dict(**default_hp_set)
        full_hp_set.update(hp_set)
        full_hp_sets.append(full_hp_set)
    return full_hp_sets


def create_hp_sets() -> list[HPSet]:
    """Create all HP sets."""
    default_hp_set: HPSet = dict(a=1, b=2, c=3)
    hp_sets: list[HPSet] = [
        dict(b=3, c=c, z=0, x=x) for c in [-1, -2] for x in [0.4, 0.5]
    ]
    full_hp_sets: list[HPSet] = to_full_hp_sets(default_hp_set, hp_sets)
    return full_hp_sets


def save_hp_set(hp_path: Path, hp_set: HPSet) -> Path:
    """Save an HP set with an index."""
    with open(hp_path, "w") as f:
        yaml.safe_dump(hp_set, f)
    return hp_path


def save_hp_sets(hp_sets: list[HPSet]) -> list[Path]:
    """Save multiple HP sets at once."""
    hps_dir = Path("hp_sets")
    if hps_dir.exists():
        shutil.rmtree(hps_dir)
    hps_dir.mkdir(parents=True)
    return [
        save_hp_set(hps_dir / f"{i}.yaml", hp_set) for i, hp_set in enumerate(hp_sets)
    ]


def hp_main(ids: dict[str, str]) -> None:
    """Create hyper-parameter sets and run jobs for them."""
    hp_sets = create_hp_sets()
    hp_paths = save_hp_sets(hp_sets)

    job_ids = [run_job(hp_path, ids) for hp_path in hp_paths]
    print(job_ids)
    time.sleep(1)


def get_exp_ids() -> dict[str, str]:
    """Validate and get ids for the current experiment."""
    repo = Repo(".")
    branch = repo.active_branch.name
    exp_prefix = "exp/"
    if not branch.startswith(exp_prefix):
        raise ValueError(
            f"Current branch must start with `{exp_prefix}`, got `{branch}`"
        )
    if repo.is_dirty(untracked_files=True):
        raise ValueError("Commit all changes before running.")
    exp_id = branch.removeprefix(exp_prefix)
    time_id = datetime.now().strftime("%Y%m%d%H%M%S")
    commit_id = repo.git.rev_parse("HEAD", short=True)
    rand_id = secrets.token_hex(3)
    return dict(exp=exp_id, time=time_id, commit=commit_id, rand=rand_id)


if __name__ == "__main__":
    exp_ids = get_exp_ids()
    hp_main(exp_ids)
