"""Generation, saving and loading of HP sets."""

import shutil
from pathlib import Path
from typing import Any

import yaml

from .git_utils import git_push_dir

HPSet = dict[str, Any]


def load_hp_set(hp_path: Path) -> HPSet:
    """Load an HP set by index."""
    with open(hp_path) as f:
        hp_set = yaml.safe_load(f)
    return hp_set


def to_full_hp_sets(default_hp_set: HPSet, hp_sets: list[HPSet]) -> list[HPSet]:
    """Convert partial HP sets to full with default values."""
    full_hp_sets = []
    for hp_set in hp_sets:
        full_hp_set = dict(**default_hp_set)
        full_hp_set.update(hp_set)
        full_hp_sets.append(full_hp_set)
    return full_hp_sets


def create_hp_sets(default_hp_set: HPSet, hp_sets: list[HPSet]) -> list[HPSet]:
    """Create all HP sets."""
    full_hp_sets: list[HPSet] = to_full_hp_sets(default_hp_set, hp_sets)
    return full_hp_sets


def save_hp_set(hp_path: Path, hp_set: HPSet) -> Path:
    """Save an HP set with an index."""
    with open(hp_path, "w") as f:
        yaml.safe_dump(hp_set, f)
    return hp_path


def save_hp_sets(hp_sets: list[HPSet], hps_dir: Path) -> list[tuple[str, Path]]:
    """Save multiple HP sets at once."""
    if hps_dir.exists():
        shutil.rmtree(hps_dir)
    hps_dir.mkdir(parents=True)
    indices = [str(i).zfill(len(str(len(hp_sets)))) for i in range(len(hp_sets))]
    return [
        (i, save_hp_set(hps_dir / f"{i}.yaml", hp_set))
        for i, hp_set in zip(indices, hp_sets, strict=True)
    ]


def prepare_hp_sets(
    default_hp_set: HPSet, hp_sets: list[HPSet], hps_dir: Path
) -> list[tuple[str, Path]]:
    """Make all preparations necessary to run with the given HP sets."""
    hp_sets = create_hp_sets(default_hp_set, hp_sets)
    hp_paths = save_hp_sets(hp_sets, hps_dir)
    git_push_dir(Path("."))
    return hp_paths
