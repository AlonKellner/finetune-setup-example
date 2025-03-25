"""Job utils."""

import os
import secrets
from collections.abc import Callable
from datetime import datetime

import git
import yaml


def get_job_ids() -> tuple[str, dict[str, str]]:
    """Validate and get ids for the current experiment."""
    job_id = os.getenv("FULL_JOB_ID")
    exp_id = os.getenv("JOB_ID_EXP")
    commit_id = os.getenv("JOB_ID_COMMIT")
    time_id = os.getenv("JOB_ID_TIME")
    rand_id = os.getenv("JOB_ID_RAND")
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
    assert job_id is not None
    assert exp_id is not None
    assert commit_id is not None
    assert time_id is not None
    assert rand_id is not None
    return job_id, dict(exp=exp_id, commit=commit_id, time=time_id, rand=rand_id)


def run_local_job(job_func: Callable) -> None:
    """Run a job locally."""
    hp_path = os.getenv("JOB_HP_PATH")
    if hp_path is None:
        hp_set = dict()
    else:
        with open(hp_path) as f:
            hp_set = yaml.safe_load(f)

    job_func(**hp_set)
