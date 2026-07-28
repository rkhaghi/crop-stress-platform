#%%
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, feature_names: list = None) -> dict:
    """
    Compute regression evaluation metrics on a held-out test set.

    Parameters
    ----------
    model         : fitted sklearn-compatible model (must have .predict).
    X_test        : np.ndarray — feature matrix.
    y_test        : np.ndarray — true target values.
    feature_names : list[str]  — used to report top feature importances (optional).

    Returns
    -------
    dict with rmse, mae, r2, and top-5 feature importances (if available).
    """
    y_pred = model.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "r2":   float(r2_score(y_test, y_pred)),
    }

    # Feature importance (XGBoost / sklearn tree models)
    if hasattr(model, "feature_importances_") and feature_names:
        importances = model.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        metrics["top_features"] = [{"feature": f, "importance": float(i)} for f, i in ranked[:10]]

    return metrics
