"""A script to upload a generated pyspy file."""

import os

import boto3
import inquirer
from types_boto3_s3.client import S3Client

from finetune_setup_example.job_utils import get_job_ids

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
            signature_version="s3v4",
            region_name=region,
            s3=dict(addressing_style="virtual"),
        ),
    )

    pyspy_path = os.getenv("PYSPY_PATH")
    if pyspy_path is None:
        raise ValueError("Env var `PYSPY_PATH` not provided.")

    job_id, ids = get_job_ids()
    exp = ids["exp"]
    commit = ids["commit"]

    bucket = "pyspy"

    response = s3_client.list_objects(Bucket=bucket, Prefix=f"{exp}-{commit}")
    if "Contents" not in response:
        print(response)
        raise RuntimeError("Invalid response from S3")

    keys = [v["Key"] for v in response["Contents"] if "Key" in v]
    keys = [k for k in keys if pyspy_path in k]
    questions = [
        inquirer.List(
            "key",
            message="Choose a key:",
            choices=keys,
        )
    ]
    answers = inquirer.prompt(questions)
    if answers is None:
        raise RuntimeError()
    key = answers["key"]
    file_path = key.removeprefix(key.split("/")[0] + "/")

    with open(file_path, "wb") as f:
        s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=f)

    print(f"Downloaded `{file_path}`.")
