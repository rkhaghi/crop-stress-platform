#%%
"""
predictor.py
------------
Load a trained model bundle and run inference on a feature vector.
"""
import os
import tempfile
import joblib
import boto3
import numpy as np


_MODEL_CACHE: dict[str, dict] = {}


def _load_model(model_path: str) -> dict:
    if model_path not in _MODEL_CACHE:
        if model_path.startswith("s3://"):
            # Download from S3 to /tmp/ then load
            parts = model_path[5:].split("/", 1)
            bucket, key = parts[0], parts[1]
            local_path = os.path.join(tempfile.gettempdir(), "model.joblib")
            boto3.client("s3").download_file(bucket, key, local_path)
            _MODEL_CACHE[model_path] = joblib.load(local_path)
        else:
            _MODEL_CACHE[model_path] = joblib.load(model_path)
    return _MODEL_CACHE[model_path]


def predict(feature_vector: dict, model_path: str | None = None) -> dict:
    """
    Run the crop-stress model on a single feature vector.

    Parameters
    ----------
    feature_vector : flat dict of feature name → numeric value
                     (output of features.build_features.build_feature_vector)
    model_path     : path to the joblib model bundle.
                     Defaults to the MODEL_PATH environment variable.

    Returns
    -------
    dict with:
        stress_index   — predicted continuous crop-stress score (0–1)
        stress_level   — categorical label: "low" | "moderate" | "high"
        feature_names  — list of features the model was trained on
    """
    path = model_path or os.environ["MODEL_PATH"]
    bundle = _load_model(path)

    model         = bundle["model"]
    feature_names = bundle["feature_names"]

    # Build ordered feature array; fill missing features with NaN
    X = np.array([feature_vector.get(f, np.nan) for f in feature_names], dtype=np.float32)
    score = float(model.predict(X.reshape(1, -1))[0])

    if score < 0.33:
        level = "low"
    elif score < 0.66:
        level = "moderate"
    else:
        level = "high"

    return {
        "stress_index": score,
        "stress_level": level,
        "feature_names": feature_names,
    }
