"""Generation, saving and loading of HP sets."""

import inspect
import shutil
from collections.abc import Callable
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


def save_hp_set(hp_path: Path, hp_set: HPSet) -> Path:
    """Save an HP set with an index."""
    with open(hp_path, "w") as f:
        yaml.safe_dump(hp_set, f)
    return hp_path


def save_hp_sets(
    full_hp_sets: list[HPSet], hp_sets: list[HPSet], hps_dir: Path
) -> list[tuple[str, Path]]:
    """Save multiple HP sets at once."""
    if hps_dir.exists():
        shutil.rmtree(hps_dir)

    diff_dir = hps_dir / "diff"
    full_dir = hps_dir / "full"
    diff_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    indices = [str(i).zfill(len(str(len(hp_sets)))) for i in range(len(hp_sets))]

    for i, hp_set in zip(indices, hp_sets, strict=True):
        save_hp_set(diff_dir / f"{i}.yaml", hp_set)

    full_paths = [
        (i, save_hp_set(full_dir / f"{i}.yaml", full_hp_set))
        for i, full_hp_set in zip(indices, full_hp_sets, strict=True)
    ]
    return full_paths


def prepare_hp_sets(
    default_hp_set: HPSet, hp_sets: list[HPSet], hps_dir: Path
) -> list[tuple[str, Path]]:
    """Make all preparations necessary to run with the given HP sets."""
    full_hp_sets = to_full_hp_sets(default_hp_set, hp_sets)
    hp_paths = save_hp_sets(full_hp_sets, hp_sets, hps_dir)
    git_push_dir(Path("."))
    return hp_paths


def get_defaults(func: Callable) -> dict[str, Any]:
    """
    Get function defaults.

    Given a function `func`, returns a dictionary where keys are parameter names
    with default values, and values are the corresponding default values.
    """
    signature = inspect.signature(func)
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
