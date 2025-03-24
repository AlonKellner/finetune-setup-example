"""A script to upload a generated pyspy file."""

import os

import boto3
from types_boto3_s3.client import S3Client

from finetune_setup_example.experiment_utils import get_exp_ids

if __name__ == "__main__":
    access_key = os.getenv("HETZNER_ACCESS_KEY")
    secret_key = os.getenv("HETZNER_SECRET_KEY")
    endpoint = os.getenv("HETZNER_ENDPOINT")
    region = os.getenv("HETZNER_REGION")

    s3_client: S3Client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        aws_session_token=None,
        config=boto3.session.Config(  # type: ignore
            retries={"max_attempts": 1, "mode": "standard"},
            signature_version="s3",
            region_name=region,
            s3=dict(addressing_style="virtual"),
        ),
    )

    job_id, ids = get_exp_ids()

    pyspy_path = os.getenv("PYSPY_PATH")
    if pyspy_path is None:
        raise ValueError("Env var `PYSPY_PATH` not provided.")

    s3_path = f"{job_id}/{pyspy_path}"
    bucket = "pyspy"

    print(f"Uploading `{bucket}/{s3_path}`...")
    with open(pyspy_path, "rb") as f:
        s3_client.upload_fileobj(Fileobj=f, Bucket=bucket, Key=s3_path)
    print("Upload successful.")
