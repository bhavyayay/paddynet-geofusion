"""
Integration tests: verify that Developer 1's schema contract,
Developer 2's models, and Developer 3's evaluation/API/dashboard all
plug into each other correctly.

Tests that need trained models or the combined dataset are skipped
(not failed) when those artifacts are absent, since they're gitignored
and must be regenerated locally. To run everything:

    cp <dataset> data/processed/combined_training_dataset.parquet
    python -m src.prediction.train
    python -m src.evaluation.evaluate_models
    python -m pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(REPO_ROOT, "data/processed/combined_training_dataset.parquet")
MODELS = [
    os.path.join(REPO_ROOT, "models/yield_random_forest.joblib"),
    os.path.join(REPO_ROOT, "models/yield_cnn_lstm.pt"),
    os.path.join(REPO_ROOT, "models/feature_scaler.joblib"),
]

needs_dataset = pytest.mark.skipif(
    not os.path.exists(DATASET),
    reason="combined_training_dataset.parquet not present (gitignored artifact)",
)
needs_models = pytest.mark.skipif(
    not all(os.path.exists(p) for p in MODELS),
    reason="trained models not present -- run 'python -m src.prediction.train'",
)


# ----------------------------------------------------------------------
# Dev 1 -> Dev 2 handoff: schema contract
# ----------------------------------------------------------------------

def test_parcel_schema_contract_is_stable():
    """Dev 2's dataset_loader depends on these fields existing in
    Dev 1's schema. If someone renames them, this test catches it
    before the pipeline breaks silently."""
    from schemas.parcel_schema import ParcelRecord, VALID_LABELS

    fields = set(ParcelRecord.__dataclass_fields__.keys())
    required_by_dev2 = {
        "parcel_id", "latitude", "longitude", "date",
        "ndvi", "ndwi", "msavi", "class_label", "class_confidence",
    }
    missing = required_by_dev2 - fields
    assert not missing, f"ParcelRecord lost fields Dev 2 depends on: {missing}"
    assert {"paddy", "non_paddy", "water"} <= set(VALID_LABELS)


# ----------------------------------------------------------------------
# Dev 2 -> Dev 3 handoff: dataset and trained models
# ----------------------------------------------------------------------

@needs_dataset
def test_training_dataset_matches_expected_schema():
    import pandas as pd
    from src.prediction.train import FEATURE_COLUMNS, TARGET_COLUMN

    df = pd.read_parquet(DATASET)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns, f"Training dataset missing column: {col}"

    usable = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    assert len(usable) > 0, "No usable training rows"
    # Yields for Haryana rice should be physically plausible
    assert usable[TARGET_COLUMN].between(0.1, 10).all()


@needs_models
def test_predict_yield_returns_plausible_values():
    from src.prediction.predict import predict_yield

    result = predict_yield(
        area_000ha=129.2, t2m_max=34.61, t2m_min=23.53,
        precip_mm=0.14, rh2m=70.44,
    )
    assert set(result.keys()) == {
        "random_forest_t_ha", "cnn_lstm_t_ha", "recommended", "recommended_model",
    }
    # Haryana rice yields are roughly 1-5 t/ha; anything wildly outside
    # that means a broken model or feature-order mismatch
    assert 0.5 < result["random_forest_t_ha"] < 8.0
    assert 0.5 < result["cnn_lstm_t_ha"] < 8.0
    assert result["recommended"] == result["random_forest_t_ha"]


# ----------------------------------------------------------------------
# Dev 3: evaluation module against real historical yields
# ----------------------------------------------------------------------

@needs_dataset
@needs_models
def test_run_evaluation_end_to_end(tmp_path, monkeypatch):
    """Full evaluation run: real dataset + real models -> report with
    sane metrics, without clobbering the checked-in artifacts."""
    monkeypatch.chdir(REPO_ROOT)
    import src.evaluation.evaluate_models as ev

    monkeypatch.setattr(ev, "REPORT_PATH", str(tmp_path / "report.json"))
    monkeypatch.setattr(ev, "PREDICTIONS_PATH", str(tmp_path / "preds.parquet"))

    report = ev.run_evaluation()

    assert report["recommended_model"] in report["models"]
    for scopes in report["models"].values():
        for m in scopes.values():
            assert m["mae_t_ha"] >= 0
            assert m["rmse_t_ha"] >= m["mae_t_ha"] - 1e-9  # RMSE >= MAE always
            assert m["r2"] <= 1.0
    # Both output artifacts written
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "preds.parquet").exists()


@needs_dataset
@needs_models
def test_evaluation_test_split_matches_dev2_training(monkeypatch):
    """The reconstructed held-out split must produce the same RF MAE
    train.py reported, proving we evaluate on truly unseen rows."""
    monkeypatch.chdir(REPO_ROOT)
    from sklearn.model_selection import train_test_split
    from src.evaluation.evaluate_models import load_dataset, reconstruct_split
    from src.prediction.train import FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE

    df = load_dataset()
    _, test_mask = reconstruct_split(df)

    # Same split computed directly the way train.py does it
    X = df[FEATURE_COLUMNS].values
    _, X_test_expected, _, _ = train_test_split(
        X, df["yield_t_ha"].values, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_test_actual = X[test_mask]
    assert sorted(map(tuple, X_test_actual.tolist())) == sorted(
        map(tuple, X_test_expected.tolist())
    )


# ----------------------------------------------------------------------
# Dev 3: API integration (in-process, no network)
# ----------------------------------------------------------------------

@needs_models
def test_api_health_predict_and_validation(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    from fastapi.testclient import TestClient
    from src.api.app import app

    with TestClient(app) as client:  # 'with' runs the startup lifespan
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["ready"] is True

        good = client.post("/predict", json={
            "area_000ha": 129.2, "t2m_max": 34.61, "t2m_min": 23.53,
            "precip_mm": 0.14, "rh2m": 70.44,
        })
        assert good.status_code == 200
        body = good.json()
        assert body["recommended_model"] == "random_forest"
        assert 0.5 < body["recommended"] < 8.0

        # Out-of-range and missing inputs must be rejected, not predicted
        bad_range = client.post("/predict", json={
            "area_000ha": -5, "t2m_max": 34.0, "t2m_min": 23.0,
            "precip_mm": 0.0, "rh2m": 70.0,
        })
        assert bad_range.status_code == 422

        missing_field = client.post("/predict", json={"area_000ha": 10.0})
        assert missing_field.status_code == 422


@needs_models
def test_api_predict_agrees_with_library_call(monkeypatch):
    """The API and the direct library interface must give identical
    numbers for identical inputs -- otherwise a consumer's choice of
    entry point would change their prediction."""
    monkeypatch.chdir(REPO_ROOT)
    from fastapi.testclient import TestClient
    from src.api.app import app
    from src.prediction.predict import predict_yield

    inputs = {"area_000ha": 60.0, "t2m_max": 35.0, "t2m_min": 24.0,
              "precip_mm": 120.0, "rh2m": 70.0}

    direct = predict_yield(**inputs)
    with TestClient(app) as client:
        via_api = client.post("/predict", json=inputs).json()

    assert np.isclose(direct["random_forest_t_ha"], via_api["random_forest_t_ha"], atol=1e-6)
    assert np.isclose(direct["cnn_lstm_t_ha"], via_api["cnn_lstm_t_ha"], atol=1e-6)


# ----------------------------------------------------------------------
# Dev 3: dashboard
# ----------------------------------------------------------------------

@needs_dataset
@needs_models
def test_dashboard_builds_valid_html(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    import src.evaluation.evaluate_models as ev
    from src.dashboard.build_dashboard import build_dashboard_html

    # Ensure evaluation artifacts exist (cheap to regenerate)
    if not (os.path.exists(ev.REPORT_PATH) and os.path.exists(ev.PREDICTIONS_PATH)):
        ev.run_evaluation()

    page = build_dashboard_html()
    assert page.startswith("<!DOCTYPE html>")
    assert "<svg" in page, "dashboard should contain at least one chart"
    assert "Model Accuracy" in page
    assert "random_forest" in page
