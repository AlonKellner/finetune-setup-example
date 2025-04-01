"""Job utils."""

import os
import secrets
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import git
import sky
import sky.jobs
import yaml

from finetune_setup_example.hp_set_handling import prepare_hp_sets


def get_job_ids() -> tuple[str, dict[str, str]]:
    """Validate and get ids for the current experiment."""
    job_id = os.getenv("FULL_JOB_ID")
    ids = {
        k.removeprefix("JOB_ID_").lower(): v
        for k, v in os.environ.items()
        if "JOB_ID_" in k
    }
    exp_id = ids.get("exp")
    commit_id = ids.get("commit")
    time_id = ids.get("time")
    rand_id = ids.get("rand")
    if any((x is None) for x in [job_id, exp_id, commit_id, time_id, rand_id]):
        repo = git.Repo(".")
        branch = repo.active_branch.name
        exp_prefix = "exp/"
        if not branch.startswith(exp_prefix):
            raise ValueError(
                f"Current branch must start with `{exp_prefix}`, got `{branch}`"
            )
        if repo.is_dirty(untracked_files=True):
            raise ValueError("Commit all changes before running.")
        exp_id = branch.removeprefix(exp_prefix)
        commit_id = repo.git.rev_parse("HEAD", short=True)
        time_id = datetime.now().strftime("%Y%m%d%H%M%S")
        rand_id = secrets.token_hex(2)
        ids = dict(
            exp=exp_id,
            commit=commit_id,
            time=time_id,
            rand=rand_id,
        )
    assert exp_id is not None
    assert commit_id is not None
    assert time_id is not None
    assert rand_id is not None
    if job_id is None:
        job_id = "-".join(ids.values())
    return job_id, ids


def run_local_job(job_func: Callable) -> None:
    """Run a job locally."""
    hp_path = os.getenv("JOB_HP_PATH")
    if hp_path is None:
        hp_set = dict()
    else:
        with open(hp_path) as f:
            hp_set = yaml.safe_load(f)

    job_func(**hp_set)


def start_jobs(hp_paths: list[tuple[str, Path]], ids: dict[str, str]) -> list[str]:
    """Run multiple SkyPilot jobs for an experiment."""
    return [
        start_job(dict(**ids, index=str(i)), str(hp_path)) for i, hp_path in hp_paths
    ]


def start_job(ids: dict[str, str], job_hp_path: str) -> str:
    """Start a SkyPilot job using the job.yaml config."""
    task = sky.Task.from_yaml("skypilot-conf/job.yaml")

    job_id = "-".join(ids.values())

    # Set environment variables dynamically using update_envs
    env_vars = {
        **{f"JOB_ID_{key.upper()}": value for key, value in ids.items()},
        "JOB_HP_PATH": job_hp_path,
        "FULL_JOB_ID": job_id,
    }
    task.update_envs(env_vars)

    # Launch the task using sky.launch
    sky.jobs.launch(task, name=job_id)
    return job_id


def start_hp_jobs(
    default_hp_set: dict[str, Any], hp_sets: list[dict[str, Any]]
) -> list[str]:
    """Start multiple SkyPilot jobs for hyper-parameter sets."""
    hp_paths = prepare_hp_sets(default_hp_set, hp_sets, Path("hp_sets"))

    _, ids = get_job_ids()

    return start_jobs(hp_paths, ids)
