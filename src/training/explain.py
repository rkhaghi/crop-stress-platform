#%%
"""
explain.py
----------
Generate SHAP-based feature importance and explanation plots for the trained model.

Usage:
    python src/training/explain.py --model models/crop_stress_model.json \
                                    --features data/features.parquet \
                                    --output reports/shap_summary.png
"""
import argparse
import pathlib

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


def explain(model_path: str, features_path: str, output_path: str | None = None) -> None:
    """
    Load model + features, compute SHAP values, and save a summary plot.

    Parameters
    ----------
    model_path    : Path to the joblib model bundle (saved by train.py).
    features_path : Path to the features parquet file.
    output_path   : Optional path to save the PNG summary plot.
    """
    bundle = joblib.load(model_path)
    model         = bundle["model"]
    feature_names = bundle["feature_names"]

    df = pd.read_parquet(features_path)
    X  = df[feature_names]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)

    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"[explain] SHAP summary saved → {output_path}")
    else:
        plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output",   default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    explain(args.model, args.features, args.output)
