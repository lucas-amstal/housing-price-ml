"""
FastAPI app that serves predictions from the trained California Housing model.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HousingFeatures, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

app = FastAPI(
    title="California Housing Price API",
    description="Predicts median house value (in $100k) for a California census block group.",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    model = get_model()
    row = pd.DataFrame([features.model_dump()])
    prediction = model.predict(row)[0]
    return PredictionResponse(predicted_median_house_value_100k=round(float(prediction), 4))
