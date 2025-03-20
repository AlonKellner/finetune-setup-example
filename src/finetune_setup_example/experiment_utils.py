"""Experiment ID generation."""

import secrets
from datetime import datetime

import git


def get_exp_ids() -> dict[str, str]:
    """Validate and get ids for the current experiment."""
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
    return dict(exp=exp_id, commit=commit_id, time=time_id, rand=rand_id)
