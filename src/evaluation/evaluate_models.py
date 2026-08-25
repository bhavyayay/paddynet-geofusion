"""
evaluate_models.py

Developer 3's evaluation module: compares the trained yield models
(Developer 2's Random Forest and CNN-LSTM) against real historical
yield numbers from the government ground-truth data.

What it does:
  1. Loads the same dataset the models were trained on
     (data/processed/combined_training_dataset.parquet), whose
     yield_t_ha column comes from real government statistics.
  2. Reconstructs the exact train/test split Developer 2 used
     (same random seed and test size), so the "held-out" numbers
     here are honest -- computed only on rows the models never saw
     during training.
  3. Runs both models over the data and computes MAE / RMSE / R2 /
     MAPE, separately for the held-out test set and the full dataset.
  4. Writes a machine-readable report to
     data/processed/evaluation_report.json and a per-row comparison
     table to data/processed/evaluation_predictions.parquet
     (actual vs predicted, per region/year) for the dashboard.

Usage:
    python -m src.evaluation.evaluate_models
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import compute_all
from src.prediction.model import build_model
from src.prediction.train import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    RF_SAVE_PATH,
    CNN_LSTM_SAVE_PATH,
    SCALER_SAVE_PATH,
)

REPORT_PATH = "data/processed/evaluation_report.json"
PREDICTIONS_PATH = "data/processed/evaluation_predictions.parquet"


def load_dataset() -> pd.DataFrame:
    """Load the combined training dataset, keeping only rows with
    complete features (matching what train.py does)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run Developer 2's data pipeline "
            "(src/prediction/combine_datasets.py) or copy the dataset "
            "into data/processed/ first."
        )
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return df


def load_trained_models():
    """Load Developer 2's saved model artifacts."""
    missing = [p for p in (RF_SAVE_PATH, CNN_LSTM_SAVE_PATH, SCALER_SAVE_PATH)
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing}. Run "
            "'python -m src.prediction.train' first."
        )
    rf_model = joblib.load(RF_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)
    cnn_lstm = build_model(n_features=len(FEATURE_COLUMNS))
    cnn_lstm.load_state_dict(torch.load(CNN_LSTM_SAVE_PATH))
    cnn_lstm.eval()
    return rf_model, cnn_lstm, scaler


def predict_batch(df: pd.DataFrame, rf_model, cnn_lstm, scaler) -> pd.DataFrame:
    """Run both models over every row, returning the frame with two
    added prediction columns."""
    X = df[FEATURE_COLUMNS].values

    rf_preds = rf_model.predict(X)

    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        dl_preds = cnn_lstm(X_t).numpy().flatten()

    out = df.copy()
    out["pred_random_forest_t_ha"] = np.round(rf_preds, 3)
    out["pred_cnn_lstm_t_ha"] = np.round(dl_preds, 3)
    out["abs_error_rf_t_ha"] = np.round(
        np.abs(out[TARGET_COLUMN] - out["pred_random_forest_t_ha"]), 3
    )
    return out


def reconstruct_split(df: pd.DataFrame):
    """
    Recreate the exact train/test split from train.py (same seed,
    same test size, same row order) so held-out metrics are honest.
    Returns boolean masks over df's rows.
    """
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    is_test = np.zeros(len(df), dtype=bool)
    is_test[test_idx] = True
    return ~is_test, is_test


def run_evaluation() -> dict:
    """Full evaluation: returns the report dict and writes artifacts."""
    df = load_dataset()
    rf_model, cnn_lstm, scaler = load_trained_models()

    results = predict_batch(df, rf_model, cnn_lstm, scaler)
    train_mask, test_mask = reconstruct_split(df)
    results["split"] = np.where(test_mask, "test", "train")

    y_true = results[TARGET_COLUMN].values
    report = {
        "dataset": DATA_PATH,
        "n_rows_evaluated": int(len(results)),
        "n_test_rows": int(test_mask.sum()),
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "models": {},
        "caveat": (
            "Held-out metrics come from a very small test set "
            f"({int(test_mask.sum())} rows). Treat them as directional. "
            "Full-dataset metrics include rows the models trained on "
            "and will look optimistic."
        ),
    }

    for name, col in [
        ("random_forest", "pred_random_forest_t_ha"),
        ("cnn_lstm", "pred_cnn_lstm_t_ha"),
    ]:
        y_pred = results[col].values
        report["models"][name] = {
            "held_out_test": compute_all(y_true[test_mask], y_pred[test_mask]),
            "full_dataset": compute_all(y_true, y_pred),
        }

    # Recommend whichever model has the lower held-out MAE
    best = min(
        report["models"],
        key=lambda m: report["models"][m]["held_out_test"]["mae_t_ha"],
    )
    report["recommended_model"] = best

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    results.to_parquet(PREDICTIONS_PATH, index=False)

    return report


def print_report(report: dict):
    print("=" * 64)
    print("MODEL EVALUATION vs REAL HISTORICAL YIELDS")
    print("=" * 64)
    print(f"Dataset: {report['dataset']}")
    print(f"Rows evaluated: {report['n_rows_evaluated']} "
          f"(held-out test rows: {report['n_test_rows']})")
    print()
    header = f"{'Model':<16}{'Scope':<16}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'MAPE%':>8}"
    print(header)
    print("-" * len(header))
    for name, scopes in report["models"].items():
        for scope_name, m in scopes.items():
            print(f"{name:<16}{scope_name:<16}"
                  f"{m['mae_t_ha']:>8.3f}{m['rmse_t_ha']:>8.3f}"
                  f"{m['r2']:>8.3f}{m['mape_pct']:>8.2f}")
    print()
    print(f"Recommended model (lowest held-out MAE): {report['recommended_model']}")
    print(f"\nCAVEAT: {report['caveat']}")
    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"Per-row comparison saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    print_report(run_evaluation())
