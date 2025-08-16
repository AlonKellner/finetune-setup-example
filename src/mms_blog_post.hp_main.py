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
            seed=seed,
            base_hf_repo=base_hf_repo,
            architecture=architecture,
            num_train_epochs=epochs,
            num_eval_steps=100,
            eval_on_start=True,
            dataloader_num_workers=20,
            fp16=True,
            push_to_hub=True,
            logging_nan_inf_filter=True,
            should_clean_groups=True,
            attn_implementation=attn_implementation,
            job_path=f"{name}.{job_type}.job.yaml",
            job_stem=name,
            job_type=job_type,
            per_device_train_batch_total_seconds=train_batch_total_seconds,
            per_device_eval_batch_total_seconds=eval_batch_total_seconds,
            hidden_dropout=dropout,
            activation_dropout=dropout,
            attention_dropout=dropout,
            final_dropout=dropout,
            layerdrop=dropout,
            sp_vocab_size=sp_vocab_size,
            sp_bpe_dropout=sp_bpe_dropout,
            pretrained_learning_rate=pretrained_learning_rate,
            adapter_learning_rate=adapter_learning_rate,
        )
        for pretrained_learning_rate in [1e-4]
        for adapter_learning_rate in [1e-3]
        for train_batch_total_seconds in [1200.0]
        for eval_batch_total_seconds in [600.0]
        for epochs in [1]
        for job_type in ["a100"]
        for dropout in [0.05]
        for sp_vocab_size, sp_bpe_dropout in [
            (32, 0.0),
            (64, 0.0),
            (128, 0.0),
            (256, 0.0),
        ]
        for base_hf_repo, architecture, attn_implementation in [
            ("facebook/w2v-bert-2.0", "w2v-bert2", "eager"),
        ]
        for seed in [42, 43]
    ]
    full_json = json.dumps(dict(enumerate(hp_sets)), indent=2)
    print(f"Created {len(hp_sets)} hyper-parameter sets:\n{full_json}")
    job_names = start_hp_jobs(default_hp_set, hp_sets, f"{name}.env")
    print(f"Started {len(job_names)} jobs: {job_names}")


if __name__ == "__main__":
    hp_main()
