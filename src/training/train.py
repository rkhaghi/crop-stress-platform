#%%
"""
train.py
--------
Train an XGBoost regressor to predict crop stress index from feature vectors.

Usage:
    python src/training/train.py --features data/features.parquet \
                                  --label stress_index \
                                  --output models/crop_stress_model.json
"""
import argparse
import pathlib

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from training.evaluate import evaluate_model

DEFAULT_PARAMS = {
    "n_estimators":     400,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
}


def train(
    features_path: str,
    label_col: str,
    output_path: str,
    test_size: float = 0.2,
) -> dict:
    """
    Load a feature parquet file, train an XGBoost model, evaluate, and save.

    Returns
    -------
    dict of evaluation metrics on the held-out test set.
    """
    df = pd.read_parquet(features_path)
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = XGBRegressor(**DEFAULT_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    metrics = evaluate_model(model, X_test, y_test, feature_names=feature_cols)
    print("[train] Test metrics:", metrics)

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_cols}, output_path)
    print(f"[train] Model saved → {output_path}")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--label",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.features, args.label, args.output, args.test_size)
