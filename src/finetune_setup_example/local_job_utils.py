"""Local job utils for running jobs locally."""

import os
from collections.abc import Callable

import yaml


def run_local_job(job_func: Callable) -> None:
    """Run a job locally."""
    hp_path = os.getenv("JOB_HP_PATH")
    if hp_path is None:
        hp_set = dict()
    else:
        with open(hp_path) as f:
            hp_set = yaml.safe_load(f)

    job_func(**hp_set, hp_set=hp_set)
