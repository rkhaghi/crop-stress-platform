"""
batch_train.py
--------------
AWS Batch entrypoint: download inputs from S3, run build + train, upload outputs.

Required environment variables:
    S3_BUCKET            — S3 bucket name
    CDSE_S3_ACCESS_KEY   — Copernicus S3 access key
    CDSE_S3_SECRET_KEY   — Copernicus S3 secret key

Optional environment variables:
    LOCATIONS_KEY        — S3 key for locations CSV  (default: data/locations.csv)
    FEATURES_KEY         — S3 key for output parquet (default: data/features.parquet)
    MODEL_KEY            — S3 key for output model   (default: models/crop_stress_model.json)
"""
import os
import subprocess
import sys

import boto3


def main() -> None:
    bucket        = os.environ["S3_BUCKET"]
    locations_key = os.environ.get("LOCATIONS_KEY", "data/locations.csv")
    features_key  = os.environ.get("FEATURES_KEY",  "data/features.parquet")
    model_key     = os.environ.get("MODEL_KEY",      "models/crop_stress_model.json")

    s3 = boto3.client("s3")

    # Step 1 — Download locations CSV from S3
    print(f"[batch] Downloading s3://{bucket}/{locations_key}")
    s3.download_file(bucket, locations_key, "/app/data/locations.csv")

    # Step 2 — Build feature dataset
    print("[batch] Running build_dataset.py ...")
    subprocess.run(
        [sys.executable, "/app/src/build_dataset.py",
         "--locations", "/app/data/locations.csv",
         "--output",    "/app/data/features.parquet"],
        check=True,
    )

    # Step 3 — Train model
    print("[batch] Running train.py ...")
    subprocess.run(
        [sys.executable, "/app/src/training/train.py",
         "--features", "/app/data/features.parquet",
         "--label",    "stress_index",
         "--output",   "/app/models/crop_stress_model.json"],
        check=True,
    )

    # Step 4 — Upload outputs to S3
    print(f"[batch] Uploading features → s3://{bucket}/{features_key}")
    s3.upload_file("/app/data/features.parquet", bucket, features_key)

    print(f"[batch] Uploading model → s3://{bucket}/{model_key}")
    s3.upload_file("/app/models/crop_stress_model.json", bucket, model_key)

    print("[batch] Done.")


if __name__ == "__main__":
    main()
