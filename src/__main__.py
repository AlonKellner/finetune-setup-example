"""A mockup of multi-HP jobs."""

from finetune_setup_example.remote_job_utils import start_hp_jobs


def hp_main() -> None:
    """Create hyper-parameter sets and run SkyPilot jobs for them."""
    default_hp_set = dict(wait_seconds=10, prefix="default")
    hp_sets = [
        *[dict(prefix=prefix) for prefix in ["Z", "Y"]],
        *[
            dict(wait_seconds=wait_seconds, prefix=prefix)
            for wait_seconds in [5, 15]
            for prefix in ["A", "B"]
        ],
    ]
    job_names = start_hp_jobs(default_hp_set, hp_sets)
    print(f"Started {len(job_names)} jobs: {job_names}")


# Example usage
if __name__ == "__main__":
    hp_main()
