"""Local job utils for running jobs locally."""

import os
from collections.abc import Callable
from pathlib import Path

import dotenv
import yaml


def run_local_job(job_func: Callable, env_file: Path | None = None) -> None:
    """Run a job locally."""
    if (env_file is not None) and env_file.exists():
        print(f"Loading environment variables from {env_file}")
        dotenv.load_dotenv(env_file)

    hp_path = os.getenv("JOB_HP_PATH")
    if hp_path is None:
        hp_set = dict()
    else:
        with open(hp_path) as f:
            hp_set = yaml.safe_load(f)

    job_func(**hp_set, hp_set=hp_set)
