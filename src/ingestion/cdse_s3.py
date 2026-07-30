from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from dotenv import load_dotenv


load_dotenv()

CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"


def create_cdse_s3_client():
    access_key = os.getenv("CDSE_S3_ACCESS_KEY")
    secret_key = os.getenv("CDSE_S3_SECRET_KEY")

    if not access_key or not secret_key:
        raise RuntimeError(
            "CDSE S3 credentials are missing. "
            "Set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY in .env."
        )

    return boto3.client(
        "s3",
        endpoint_url=CDSE_S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    parsed = urlparse(s3_url)

    # Accept s3:// or HTTPS eodata URLs
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
    elif parsed.scheme in ("http", "https") and "eodata.dataspace.copernicus.eu" in parsed.netloc:
        bucket = "eodata"
        key = parsed.path.lstrip("/")
    else:
        raise ValueError(f"Unsupported URL scheme or host: {s3_url}")

    if not bucket or not key:
        raise ValueError(f"Invalid URL: {s3_url}")

    return bucket, key


def download_asset(
    s3_url: str,
    output_path: str | Path,
) -> Path:
    bucket, key = parse_s3_url(s3_url)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    client = create_cdse_s3_client()

    print(f"[cdse_s3] Downloading s3://{bucket}/{key}")

    client.download_file(
        Bucket=bucket,
        Key=key,
        Filename=str(destination),
    )

    return destination