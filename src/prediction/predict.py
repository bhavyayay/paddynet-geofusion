"""
predict.py

Clean prediction interface for the trained yield models. This is the
file Developer 3 should import from when building the API/dashboard --
not train.py, which is for training/evaluation, not inference.

RECOMMENDED MODEL: Random Forest.
It genuinely outperforms the CNN-LSTM at this dataset size (R2 0.157
vs ~0.01 -- see src/prediction/README.md for full results). Both are
provided here since the team brief calls for a deep learning model,
but Random Forest's predictions should be treated as more reliable.

REQUIRES: run `python -m src.prediction.train` at least once first,
to generate the saved model files in models/.

Usage (as a library):
    from src.prediction.predict import predict_yield

    result = predict_yield(
        area_000ha=120.0,
        t2m_max=34.0,
        t2m_min=24.0,
        precip_mm=0.5,
        rh2m=72.0,
    )
    print(result)
    # {'random_forest_t_ha': 3.42, 'cnn_lstm_t_ha': 3.10, 'recommended': 3.42}

Usage (from command line, for a quick manual check):
    python -m src.prediction.predict
"""

import joblib
import torch

from src.prediction.model import build_model

RF_MODEL_PATH = "models/yield_random_forest.joblib"
CNN_LSTM_MODEL_PATH = "models/yield_cnn_lstm.pt"
SCALER_PATH = "models/feature_scaler.joblib"

# Must match the exact order used in train.py's FEATURE_COLUMNS
FEATURE_ORDER = ["area_000ha", "t2m_max", "t2m_min", "precip_mm", "rh2m"]


def _load_models():
    """
    Loads all saved model artifacts. Raises a clear error if
    train.py hasn't been run yet, rather than a cryptic file-not-found
    trace.
    """
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Could not find saved model files. Run "
            "'python -m src.prediction.train' first to train and save "
            "the models before calling predict_yield()."
        ) from e

    cnn_lstm_model = build_model(n_features=len(FEATURE_ORDER))
    cnn_lstm_model.load_state_dict(torch.load(CNN_LSTM_MODEL_PATH))
    cnn_lstm_model.eval()

    return rf_model, cnn_lstm_model, scaler


def predict_yield(area_000ha: float, t2m_max: float, t2m_min: float,
                   precip_mm: float, rh2m: float) -> dict:
    """
    Predict rice yield (tonnes/hectare) from area and weather features.

    Args:
        area_000ha: Rice-cultivated area, thousand hectares
        t2m_max: Max daily temperature, degrees C
        t2m_min: Min daily temperature, degrees C
        precip_mm: Daily precipitation, mm
        rh2m: Relative humidity, percent

    Returns:
        dict with keys:
          - random_forest_t_ha: Random Forest's prediction
          - cnn_lstm_t_ha: CNN-LSTM's prediction
          - recommended: the Random Forest prediction (more reliable
            at this dataset size -- see README for why)
          - recommended_model: which model 'recommended' came from,
            for display purposes
    """
    rf_model, cnn_lstm_model, scaler = _load_models()

    features = [[area_000ha, t2m_max, t2m_min, precip_mm, rh2m]]

    # Random Forest -- no scaling needed, tree-based models don't
    # require feature normalization
    rf_pred = float(rf_model.predict(features)[0])

    # CNN-LSTM -- needs the same scaling used during training
    features_scaled = scaler.transform(features)
    features_t = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        cnn_lstm_pred = float(cnn_lstm_model(features_t).item())

    return {
        "random_forest_t_ha": round(rf_pred, 3),
        "cnn_lstm_t_ha": round(cnn_lstm_pred, 3),
        "recommended": round(rf_pred, 3),
        "recommended_model": "random_forest",
    }


if __name__ == "__main__":
    # Quick manual check using Fatehabad-like conditions as an example
    print("Example prediction (Fatehabad-like conditions):")
    result = predict_yield(
        area_000ha=129.2,
        t2m_max=34.61,
        t2m_min=23.53,
        precip_mm=0.14,
        rh2m=70.44,
    )
    print(result)
    print()
    print("For reference, Fatehabad's actual 2021-22 recorded yield was 4.057 t/ha.")
