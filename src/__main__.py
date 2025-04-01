"""A mockup of multi-HP jobs."""

from pathlib import Path

import sky
import sky.jobs

from finetune_setup_example.hp_set_handling import prepare_hp_sets
from finetune_setup_example.job_utils import get_job_ids


def start_job(ids: dict[str, str], job_hp_path: str) -> None:
    """Start a SkyPilot job using the finetune_setup_example.yaml config."""
    task = sky.Task.from_yaml("skypilot/finetune_setup_example.yaml")

    # Set environment variables dynamically using update_envs
    env_vars = {
        **{f"JOB_ID_{key.upper()}": value for key, value in ids.items()},
        "JOB_HP_PATH": job_hp_path,
    }
    task.update_envs(env_vars)

    # Launch the task using sky.launch
    sky.jobs.launch(task)


def start_jobs(hp_paths: list[tuple[str, Path]], ids: dict[str, str]) -> None:
    """Run multiple SkyPilot jobs for an experiment."""
    for i, hp_path in hp_paths:
        start_job(dict(**ids, index=str(i)), str(hp_path))


def hp_main() -> None:
    """Create hyper-parameter sets and run SkyPilot jobs for them."""
    default_hp_set = dict(a=10, b=2, c=30)
    hp_sets = [dict(b=3, c=c, z=0, x=x) for c in [-1, -2] for x in [0.4, 0.5]]
    hp_paths = prepare_hp_sets(default_hp_set, hp_sets, Path("hp_sets"))

    _, ids = get_job_ids()

    start_jobs(hp_paths, ids)


# Example usage
if __name__ == "__main__":
    hp_main()
