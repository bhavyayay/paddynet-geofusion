"""
metrics.py

Standard regression accuracy measurements used across the evaluation
module. Kept as pure functions on numpy arrays so they're trivially
testable and reusable by the API and dashboard.

All yield values are in tonnes per hectare (t/ha).
"""

import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error: average size of the prediction error."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error: like MAE but punishes big misses harder."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    """
    R-squared: fraction of the yield variation the model explains.
    1.0 is perfect, 0.0 means no better than predicting the mean,
    negative means worse than predicting the mean.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        # All true values identical -- R2 is undefined; report 0.
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mape(y_true, y_pred) -> float:
    """
    Mean Absolute Percentage Error, in percent. Rows where the true
    value is 0 are excluded (division by zero).
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_all(y_true, y_pred) -> dict:
    """All metrics in one dict, rounded for reporting."""
    return {
        "mae_t_ha": round(mae(y_true, y_pred), 4),
        "rmse_t_ha": round(rmse(y_true, y_pred), 4),
        "r2": round(r2(y_true, y_pred), 4),
        "mape_pct": round(mape(y_true, y_pred), 2),
        "n_samples": int(len(np.asarray(y_true))),
    }


def _as_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {y_true.shape} vs {y_pred.shape}"
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    return y_true, y_pred
