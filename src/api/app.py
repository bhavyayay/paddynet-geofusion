"""
app.py

Developer 3's prediction API: a small FastAPI service that exposes
Developer 2's trained yield models to other programs.

Endpoints:
  GET  /health     -- liveness check + whether model artifacts are loaded
  POST /predict    -- yield prediction from area + weather features
  GET  /metrics    -- latest evaluation report (accuracy vs real yields)
  GET  /dashboard  -- human-friendly HTML dashboard
  GET  /docs       -- interactive API docs (provided by FastAPI)

Run (from the repo root, with the venv active):
    uvicorn src.api.app:app --reload

NOTE: this service has NO authentication. It is intended for local
development and team demos only. Put it behind proper auth before
exposing it anywhere beyond localhost.

Model caching: unlike src.prediction.predict.predict_yield (which
reloads model files on every call), this service loads the artifacts
once at startup and reuses them for every request.
"""

import json
import os
from contextlib import asynccontextmanager

import joblib
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.prediction.model import build_model
from src.prediction.predict import (
    RF_MODEL_PATH,
    CNN_LSTM_MODEL_PATH,
    SCALER_PATH,
    FEATURE_ORDER,
)

EVALUATION_REPORT_PATH = "data/processed/evaluation_report.json"

# Loaded once at startup, reused per request
_models = {"rf": None, "cnn_lstm": None, "scaler": None}


def load_models_into_cache():
    """Load model artifacts once. Raises FileNotFoundError with a
    helpful message if training hasn't been run yet."""
    missing = [p for p in (RF_MODEL_PATH, CNN_LSTM_MODEL_PATH, SCALER_PATH)
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing}. Run "
            "'python -m src.prediction.train' from the repo root first."
        )
    _models["rf"] = joblib.load(RF_MODEL_PATH)
    _models["scaler"] = joblib.load(SCALER_PATH)
    cnn_lstm = build_model(n_features=len(FEATURE_ORDER))
    cnn_lstm.load_state_dict(torch.load(CNN_LSTM_MODEL_PATH))
    cnn_lstm.eval()
    _models["cnn_lstm"] = cnn_lstm


def models_loaded() -> bool:
    return all(v is not None for v in _models.values())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup; if artifacts are missing we still start
    # (so /health can report the problem) but /predict will 503.
    try:
        load_models_into_cache()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
    yield


app = FastAPI(
    title="PaddyNet-GeoFusion Yield Prediction API",
    description=(
        "Predicts rice yield (t/ha) for Haryana from cultivated area "
        "and weather features, using models trained on government "
        "yield statistics and NASA POWER weather data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Input features, in the same units the model was trained on."""
    area_000ha: float = Field(..., ge=0, le=100000,
                              description="Rice-cultivated area, thousand hectares")
    t2m_max: float = Field(..., ge=-20, le=60,
                           description="Max daily temperature, degrees C")
    t2m_min: float = Field(..., ge=-40, le=50,
                           description="Min daily temperature, degrees C")
    precip_mm: float = Field(..., ge=0, le=1000,
                             description="Daily precipitation, mm")
    rh2m: float = Field(..., ge=0, le=100,
                        description="Relative humidity, percent")


class PredictionResponse(BaseModel):
    random_forest_t_ha: float
    cnn_lstm_t_ha: float
    recommended: float
    recommended_model: str


@app.get("/health")
def health():
    """Liveness check. 'ready' is true only when models are loaded."""
    return {"status": "ok", "ready": models_loaded()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict rice yield (t/ha) from area and weather features."""
    if not models_loaded():
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run 'python -m src.prediction.train' "
                   "and restart the API.",
        )

    features = [[getattr(request, name) for name in FEATURE_ORDER]]

    rf_pred = float(_models["rf"].predict(features)[0])

    features_scaled = _models["scaler"].transform(features)
    features_t = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        cnn_lstm_pred = float(_models["cnn_lstm"](features_t).item())

    return PredictionResponse(
        random_forest_t_ha=round(rf_pred, 3),
        cnn_lstm_t_ha=round(cnn_lstm_pred, 3),
        recommended=round(rf_pred, 3),
        recommended_model="random_forest",
    )


@app.get("/metrics")
def metrics():
    """Latest model evaluation report (accuracy vs real historical
    yields). Generated by 'python -m src.evaluation.evaluate_models'."""
    if not os.path.exists(EVALUATION_REPORT_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation report found at {EVALUATION_REPORT_PATH}. "
                   "Run 'python -m src.evaluation.evaluate_models' first.",
        )
    with open(EVALUATION_REPORT_PATH) as f:
        return json.load(f)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Human-friendly dashboard: charts of predicted vs actual yield,
    evaluation metrics, and project status."""
    from src.dashboard.build_dashboard import build_dashboard_html
    try:
        return build_dashboard_html()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
