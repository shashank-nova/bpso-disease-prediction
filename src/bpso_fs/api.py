"""Optional FastAPI inference service. Set BPSO_MODEL to a trained artifact path."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .deployment import load_artifact, prediction_frame

app = FastAPI(title="BPSO Disease Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    records: list[dict[str, object]] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def _artifact():
    path = os.environ.get("BPSO_MODEL", "models/disease_model.joblib")
    return load_artifact(path)


@app.get("/health")
def health():
    try:
        _artifact()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable") from exc
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    try:
        return _artifact()["metadata"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable") from exc


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        artifact = _artifact()
        frame = prediction_frame(request.records, artifact["metadata"]["features"])
        model = artifact["pipeline"]
        labels = model.predict(frame).astype(int)
        probabilities = model.predict_proba(frame)[:, 1]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Prediction failed") from exc
    return {
        "predictions": [
            {"label": int(label), "probability": float(prob)}
            for label, prob in zip(labels, probabilities, strict=True)
        ]
    }
