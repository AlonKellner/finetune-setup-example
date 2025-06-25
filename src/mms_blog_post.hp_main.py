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
        dict(
            base_hf_repo="facebook/mms-1b-all",
            should_freeze_base_model=False,
            fp16=True,
            attn_implementation="flash_attention_2",
            torch_compile=False,
            num_train_epochs=3,
            per_device_train_batch_total_seconds=batch_total_seconds,
            per_device_eval_batch_total_seconds=batch_total_seconds,
            dataloader_num_workers=16,
        )
        for batch_total_seconds in [1200.0]
    ]
    full_json = json.dumps(dict(enumerate(hp_sets)), indent=2)
    print(f"Created {len(hp_sets)} hyper-parameter sets:\n{full_json}")
    job_names = start_hp_jobs(
        f"{name}.job.yaml", default_hp_set, hp_sets, f"{name}.env"
    )
    print(f"Started {len(job_names)} jobs: {job_names}")


if __name__ == "__main__":
    hp_main()
