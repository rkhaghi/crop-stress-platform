#%%
"""
train.py
--------
Train an XGBoost regressor to predict crop stress index from feature vectors.

Usage:
    python src/training/train.py --features data/features.parquet --label stress_index  --output models/crop_stress_model.json
"""
import argparse
import os
import pathlib
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import joblib
import pandas as pd
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

    Uses a temporal split (earliest 80% → train, latest 20% → test) to
    prevent future data leaking into training.

    Returns
    -------
    dict of evaluation metrics on the held-out test set.
    """
    df = pd.read_parquet(features_path)

    # Temporal split — sort by date if present, otherwise fall back to row order
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    exclude_cols = {label_col, "date"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[label_col].values

    print(f"[train] Train: {len(train_df)} samples | Test: {len(test_df)} samples")

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
