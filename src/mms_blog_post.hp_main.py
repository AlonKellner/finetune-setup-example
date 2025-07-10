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
    default_hp_set["job_path"] = f"{name}.job.yaml"
    hp_sets = [
        dict(
            job_path=f"{name}.{job_type}.job.yaml",
            job_stem=name,
            job_type=job_type,
            base_hf_repo="facebook/mms-1b-all",
            should_freeze_base_model=False,
            fp16=True,
            attn_implementation="flash_attention_2",
            num_train_epochs=5,
            dataloader_num_workers=16,
            effective_batch_size=128,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=256,
            per_device_train_batch_total_seconds=batch_total_seconds,
            per_device_eval_batch_total_seconds=batch_total_seconds,
            hidden_dropout=dropout,
            activation_dropout=dropout,
            attention_dropout=dropout,
            final_dropout=dropout,
            layerdrop=dropout,
        )
        for batch_total_seconds in [2100.0]
        for job_type in ["a100"]
        for dropout in [0.0, 0.01, 0.02]
    ]
    full_json = json.dumps(dict(enumerate(hp_sets)), indent=2)
    print(f"Created {len(hp_sets)} hyper-parameter sets:\n{full_json}")
    job_names = start_hp_jobs(default_hp_set, hp_sets, f"{name}.env")
    print(f"Started {len(job_names)} jobs: {job_names}")


if __name__ == "__main__":
    hp_main()
