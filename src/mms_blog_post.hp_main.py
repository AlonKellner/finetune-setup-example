"""A real example of multi-HP jobs."""

import json

from finetune_setup_example.hp_set_handling import get_defaults
from finetune_setup_example.mains.mms_common_voice_adaptune import main
from finetune_setup_example.remote_job_utils import start_hp_jobs


def hp_main() -> None:
    """Create hyper-parameter sets and run SkyPilot jobs for them."""
    name = "src/mms_blog_post"
    print(f"Starting hyper-parameter jobs for {name}...")
    default_hp_set = get_defaults(main)
    hp_sets = [
        dict(base_hf_repo=base_hf_repo)
        for base_hf_repo in [
            "facebook/mms-1b-all",
            "facebook/mms-1b-l1107",
            "facebook/mms-1b-fl102",
            "facebook/mms-1b",
            "facebook/mms-300m",
            "mms-meta/mms-zeroshot-300m",
        ]
    ]
    full_json = json.dumps(dict(enumerate(hp_sets)), indent=2)
    print(f"Created {len(hp_sets)} hyper-parameter sets:\n{full_json}")
    job_names = start_hp_jobs(
        f"{name}.job.yaml", default_hp_set, hp_sets, f"{name}.env"
    )
    print(f"Started {len(job_names)} jobs: {job_names}")


if __name__ == "__main__":
    hp_main()
