import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "model.joblib"


@pytest.fixture(scope="module", autouse=True)
def ensure_model_exists():
    """Train a tiny model once so the API has something to load, unless one
    already exists (e.g. from running `python src/train.py` beforehand)."""
    if not MODEL_PATH.exists():
        import mlflow

        mlflow.set_tracking_uri(f"file://{ROOT}/tests/.tmp_mlruns")
        from train import train

        train(n_estimators=10, max_depth=2)
    yield


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_a_number(client):
    payload = {
        "MedInc": 8.3,
        "HouseAge": 41,
        "AveRooms": 6.98,
        "AveBedrms": 1.02,
        "Population": 322,
        "AveOccup": 2.55,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_median_house_value_100k" in body
    assert isinstance(body["predicted_median_house_value_100k"], float)


def test_predict_rejects_missing_fields(client):
    response = client.post("/predict", json={"MedInc": 8.3})
    assert response.status_code == 422
