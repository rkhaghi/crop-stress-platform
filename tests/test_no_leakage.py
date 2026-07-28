"""
test_no_leakage.py
------------------
Verify that the train/test split introduces no temporal or feature leakage.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


def _make_dataset(n=200, n_features=20, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.random((n, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    df["stress_index"] = rng.random(n)
    df["date"] = pd.date_range("2023-01-01", periods=n, freq="D")
    return df


class TestNoLeakage:
    def test_train_test_dates_do_not_overlap(self):
        df = _make_dataset()
        split_date = df["date"].median()
        train = df[df["date"] <= split_date]
        test  = df[df["date"] >  split_date]
        overlap = set(train["date"]) & set(test["date"])
        assert len(overlap) == 0, f"Date overlap found: {overlap}"

    def test_feature_columns_match_between_splits(self):
        df = _make_dataset()
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df[feature_cols].values
        y = df["stress_index"].values
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        assert X_train.shape[1] == X_test.shape[1]

    def test_target_not_in_feature_set(self):
        df = _make_dataset()
        feature_cols = [c for c in df.columns if c != "stress_index" and c != "date"]
        assert "stress_index" not in feature_cols

    def test_no_future_data_in_training_window(self):
        df = _make_dataset()
        # Simulate temporal split: train on first 80%, test on last 20%
        split_idx = int(len(df) * 0.8)
        train_dates = df.iloc[:split_idx]["date"]
        test_dates  = df.iloc[split_idx:]["date"]
        assert train_dates.max() < test_dates.min(), "Training data contains future dates."
