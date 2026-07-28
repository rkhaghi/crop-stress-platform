#%%
"""
predictor.py
------------
Load a trained model bundle and run inference on a feature vector.
"""
import os
import joblib
import numpy as np


_MODEL_BUNDLE = None  # module-level cache so Lambda container reuses it


def _load_model(model_path: str) -> dict:
    """Load model bundle from disk (cached after first call)."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = joblib.load(model_path)
    return _MODEL_BUNDLE


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
