"""Job utils."""

import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import git
from dotenv import dotenv_values

from .hp_set_handling import prepare_hp_sets


def get_job_ids(allow_dirty: bool = False) -> tuple[str, dict[str, str]]:
    """Validate and get ids for the current experiment."""
    job_id = os.getenv("JOB_FULL_ID")
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
        if (not allow_dirty) and repo.is_dirty(untracked_files=True):
            raise ValueError("Commit all changes before running.")
        exp_id = branch.removeprefix(exp_prefix)
        commit_id = repo.git.rev_parse("HEAD", short=True)
        time_id = datetime.now().strftime("%Y%m%dT%H%M%S")
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


def start_jobs(
    job_yaml_paths: list[str],
    hp_paths: list[tuple[str, Path]],
    ids: dict[str, str],
    env_file: str | None = None,
) -> list[str]:
    """Run multiple SkyPilot jobs for an experiment."""
    return [
        start_job(job_path, dict(**ids, index=str(i)), str(hp_path), env_file=env_file)
        for (i, hp_path), job_path in zip(hp_paths, job_yaml_paths, strict=True)
    ]


def start_job(
    job_yaml_path: str,
    ids: dict[str, str],
    job_hp_path: str,
    env_file: str | None = None,
) -> str:
    """Start a SkyPilot job using the job.yaml config."""
    import sky  # noqa: PLC0415
    import sky.jobs  # noqa: PLC0415

    print(f"\nStarting job with hp path: {job_hp_path}")

    job_id = "-".join(ids.values())
    print(f"Using job id: {job_id}")

    # Set environment variables dynamically using update_envs
    env_vars: dict[str, str] = {
        **{f"JOB_ID_{key.upper()}": value for key, value in ids.items()},
        "JOB_HP_PATH": job_hp_path,
        "JOB_FULL_ID": job_id,
    }
    if env_file is not None:
        print(f"Loading environment variables from {env_file}")
        dot_vars = dotenv_values(env_file)
        dot_vars = {k: v for k, v in dot_vars.items() if v is not None}
        env_vars.update(dot_vars)

    os.environ.update(env_vars)

    print(f"Loading job configuration with id: {job_id}")
    task = sky.Task.from_yaml(job_yaml_path)
    task.name = job_id

    # Launch the task using sky.launch
    print(f"Launching job with id: {job_id}")
    sky_id = sky.jobs.launch(task, name=job_id)
    print(f"Job started successfully with id: {job_id}")
    print(f"SkyPilot job ID: {sky_id}")
    return job_id


def start_hp_jobs(
    default_hp_set: dict[str, Any],
    hp_sets: list[dict[str, Any]],
    env_file: str | None = None,
) -> list[str]:
    """Start multiple SkyPilot jobs for hyper-parameter sets."""
    hp_paths = prepare_hp_sets(default_hp_set, hp_sets, Path("hp_sets"))

    _, ids = get_job_ids()
    print(f"Using job ids: {ids}")

    return start_jobs(
        [hp["job_path"] for hp in hp_sets], hp_paths, ids, env_file=env_file
    )
