"""Git utilities."""

from pathlib import Path

import git


def git_checkout(commit_hash: str) -> None:
    """Checkout code by commit hash."""
    repo = git.Repo(".")
    repo.git.checkout(commit_hash)


def git_push_dir(dir_to_push: Path) -> None:
    """Push changes from a specific directory."""
    repo = git.Repo(".")
    if repo.is_dirty(untracked_files=True):
        print("Auto committing changes...")
        repo.git.add(str(dir_to_push))
        commit_message = f"auto: {dir_to_push}"
        repo.index.commit(commit_message)
        origin = repo.remote(name="origin")
        origin.push()
        print("Auto commit done.")
