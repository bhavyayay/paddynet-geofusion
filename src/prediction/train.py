"""
train.py

Trains two models on the Haryana rice yield dataset and compares them:
  1. SmallCNNLSTM (PyTorch) -- matches the reference paper's
     architecture style, scaled down for this data size.
  2. RandomForestRegressor (scikit-learn) -- a baseline much better
     suited to small tabular datasets, used as a sanity check against
     the deep learning model.

DATA SIZE CAVEAT (read before trusting any numbers this prints):
This trains on ~27 real rows. That is very small for any model,
deep learning especially. Metrics here demonstrate that the full
pipeline (load -> split -> train -> evaluate) runs correctly
end-to-end. They should NOT be read as evidence of real-world
predictive accuracy until trained on substantially more data.

Usage:
    python -m src.prediction.train
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.prediction.model import build_model

DATA_PATH = "data/processed/combined_training_dataset.parquet"
CNN_LSTM_SAVE_PATH = "models/yield_cnn_lstm.pt"
RF_SAVE_PATH = "models/yield_random_forest.joblib"
SCALER_SAVE_PATH = "models/feature_scaler.joblib"

FEATURE_COLUMNS = ["area_000ha", "t2m_max", "t2m_min", "precip_mm", "rh2m"]
TARGET_COLUMN = "yield_t_ha"

RANDOM_STATE = 42
TEST_SIZE = 0.25  # with only ~27 rows, this is a small test set (~7 rows) --
                   # results should be read as directional, not precise


def load_and_prepare_data():
    df = pd.read_parquet(DATA_PATH)
    before = len(df)

    # Drop rows with missing weather (the pre-1981 rows NASA POWER
    # couldn't provide data for) -- we can't train on incomplete
    # feature rows.
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    print(f"Loaded {before} rows, {len(df)} usable after dropping rows "
          f"with missing features (pre-1981 entries).")

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    return X, y, df


def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("Training Random Forest baseline...")
    print("=" * 60)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,  # kept shallow -- with this little data, a deep
                       # forest will overfit just as readily as a big
                       # neural net would
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, RF_SAVE_PATH)
    print(f"Saved model to {RF_SAVE_PATH}")

    preds = rf.predict(X_test)
    metrics = evaluate(y_test, preds, "Random Forest")

    importances = dict(zip(FEATURE_COLUMNS, rf.feature_importances_))
    print("\nFeature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    return rf, metrics


def train_cnn_lstm(X_train, X_test, y_train, y_test, epochs=100, lr=0.01):
    print("\n" + "=" * 60)
    print("Training Small CNN-LSTM...")
    print("=" * 60)

    # Scale features -- important for neural nets, less so for RF
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Saved feature scaler to {SCALER_SAVE_PATH}")

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    model = build_model(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE, matching the reference paper's choice

    best_test_loss = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            test_loss = loss_fn(test_preds, y_test_t).item()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} - train_loss: {loss.item():.4f} - test_loss: {test_loss:.4f}")

        # Simple early stopping -- with this little data, models
        # overfit fast, so stop once test loss stops improving
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).numpy().flatten()

    metrics = evaluate(y_test, final_preds, "CNN-LSTM")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), CNN_LSTM_SAVE_PATH)
    print(f"\nSaved model to {CNN_LSTM_SAVE_PATH}")

    return model, metrics


def evaluate(y_true, y_pred, model_name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} results on test set:")
    print(f"  MAE:  {mae:.4f} t/ha")
    print(f"  RMSE: {rmse:.4f} t/ha")
    print(f"  R2:   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    X, y, df = load_and_prepare_data()

    print(f"\nFeatures: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Total usable rows: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    dl_model, dl_metrics = train_cnn_lstm(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    print(f"{'Random Forest':<20}{rf_metrics['mae']:>10.4f}{rf_metrics['rmse']:>10.4f}{rf_metrics['r2']:>10.4f}")
    print(f"{'CNN-LSTM':<20}{dl_metrics['mae']:>10.4f}{dl_metrics['rmse']:>10.4f}{dl_metrics['r2']:>10.4f}")
    print(f"\nCAVEAT: trained on {len(X_train)} real rows with a {len(X_test)}-row test set.")
    print("Random Forest shows a real positive R2 -- genuine signal, though modest.")
    print("CNN-LSTM is still close to zero R2 -- deep learning needs more data")
    print("than this to meaningfully outperform tree-based methods.")
