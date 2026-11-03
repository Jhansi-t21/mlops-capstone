"""Step 7 - MODEL DEPLOYMENT with FastAPI.

Endpoints
    GET  /health          liveness + "is the model loaded?"  (used by Docker and CI)
    GET  /model-info      which registered model version is being served
    POST /predict         one patient record -> prediction + probability
    POST /predict/batch   many records at once

Every prediction request is appended to logs/predictions.csv. Evidently reads that file
(python -m src.monitor --current logs/predictions.csv) to check for data drift.

Run locally:   uvicorn app.main:app --reload
Swagger UI:    http://127.0.0.1:8000/docs
"""
import csv
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.config import FEATURE_COLUMNS, LABELS  # also makes FeatureEngineer importable for joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
MODEL_INFO_PATH = Path(os.getenv("MODEL_INFO_PATH", ROOT / "models" / "model_info.json"))
PREDICTION_LOG = Path(os.getenv("PREDICTION_LOG", ROOT / "logs" / "predictions.csv"))
SAMPLES_PATH = Path(__file__).with_name("sample_payloads.json")

_log_lock = threading.Lock()


def _load_example() -> dict:
    try:
        return json.loads(SAMPLES_PATH.read_text())["malignant_example"]
    except Exception:
        return {c: 0.0 for c in FEATURE_COLUMNS}


# ---- request / response schemas -------------------------------------------------------
# The 30 input fields are generated from FEATURE_COLUMNS so there is one source of truth.
PredictionRequest = create_model(
    "PredictionRequest",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": _load_example()}),
    **{col: (float, Field(..., ge=0, description=f"Tumour measurement: {col}"))
       for col in FEATURE_COLUMNS},
)


class PredictionResponse(BaseModel):
    prediction: int = Field(description="1 = malignant, 0 = benign")
    label: str
    probability_malignant: float
    model_version: str


class BatchRequest(BaseModel):
    records: list[PredictionRequest] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


# ---- app state ------------------------------------------------------------------------
state: dict = {"model": None, "info": {}}


def load_model() -> None:
    """Load model + metadata. If the model file is missing the API still starts,
    /health reports model_loaded=false and /predict answers 503."""
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["info"] = (json.loads(MODEL_INFO_PATH.read_text())
                         if MODEL_INFO_PATH.exists() else {})
    else:
        state["model"], state["info"] = None, {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="MLOps capstone: serves the champion model from the MLflow registry.",
    version="1.0.0",
    lifespan=lifespan,
)


def _model_version() -> str:
    info = state["info"]
    if not info:
        return "unknown"
    return f"{info.get('registered_model_name')}/v{info.get('version')}"


def _predict(df: pd.DataFrame) -> list[PredictionResponse]:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train it first.")
    df = df[FEATURE_COLUMNS]
    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)
    _log_predictions(df, preds, proba)
    return [
        PredictionResponse(prediction=int(p), label=LABELS[int(p)],
                           probability_malignant=round(float(pr), 4),
                           model_version=_model_version())
        for p, pr in zip(preds, proba)
    ]


def _log_predictions(df: pd.DataFrame, preds, proba) -> None:
    """Append raw inputs + outputs to a CSV for later drift monitoring."""
    try:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock:
            new_file = not PREDICTION_LOG.exists()
            with open(PREDICTION_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["timestamp", *FEATURE_COLUMNS, "prediction", "probability"])
                now = datetime.now(timezone.utc).isoformat()
                for row, p, pr in zip(df.itertuples(index=False), preds, proba):
                    writer.writerow([now, *row, int(p), round(float(pr), 4)])
    except OSError:
        pass  # logging must never break a prediction


# ---- endpoints ------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "breast-cancer-prediction-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None,
            "model_version": _model_version()}


@app.get("/model-info")
def model_info():
    if not state["info"]:
        raise HTTPException(status_code=503, detail="No model metadata available.")
    return {k: v for k, v in state["info"].items() if k != "feature_columns"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):  # type: ignore[valid-type]
    return _predict(pd.DataFrame([request.model_dump()]))[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    df = pd.DataFrame([r.model_dump() for r in request.records])
    return BatchResponse(predictions=_predict(df))
