"""S3 utilities."""

import os

import boto3
from types_boto3_s3 import S3Client


def create_s3_client() -> tuple[S3Client, S3Client]:
    """Create an S3 client."""
    access_key = os.getenv("HETZNER_ACCESS_KEY")
    secret_key = os.getenv("HETZNER_SECRET_KEY")
    endpoint = os.getenv("HETZNER_ENDPOINT")
    region = os.getenv("HETZNER_REGION")
    s3_client, s3_client_v2 = [
        boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            aws_session_token=None,
            config=boto3.session.Config(  # type: ignore
                retries={"max_attempts": 1, "mode": "standard"},
                signature_version=signature_version,
                region_name=region,
                s3=dict(addressing_style="virtual"),
            ),
        )
        for signature_version in ["s3v4", "s3"]
    ]

    return s3_client, s3_client_v2
