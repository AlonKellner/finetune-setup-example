"""A real example of multi-HP jobs."""

from finetune_setup_example.hp_set_handling import get_defaults
from finetune_setup_example.mains.mms_common_voice_adaptune import main
from finetune_setup_example.remote_job_utils import start_hp_jobs


def hp_main() -> None:
    """Create hyper-parameter sets and run SkyPilot jobs for them."""
    name = "mms_blog_post"
    default_hp_set = get_defaults(main)
    hp_sets = [
        dict(num_training_steps=num_training_steps, num_train_epochs=None)
        for num_training_steps in [100, 200, 300]
    ]
    job_names = start_hp_jobs(
        f"{name}.job.yaml", default_hp_set, hp_sets, f"{name}.env"
    )
    print(f"Started {len(job_names)} jobs: {job_names}")


if __name__ == "__main__":
    hp_main()
