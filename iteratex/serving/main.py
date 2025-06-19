"""FastAPI inference server.

Currently uses a dummy model that always predicts 0.
Will later load `registry/production/model.pkl`.
"""
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, RootModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

PREDICTIONS_TOTAL = Counter("iteratex_predictions_total", "Total predictions served")
REQUEST_LATENCY = Histogram(
    "iteratex_request_latency_seconds",
    "Prediction request latency",
)
MODEL_VERSION_GAUGE = Gauge("iteratex_model_version_info", "Current model version", ['version'])

app = FastAPI(title="IteraTex Inference API", version="0.0.1")


class PredictRequest(RootModel[Dict[str, Any]]):
    """Incoming feature map. Uses Pydantic v2 RootModel."""


class PredictResponse(BaseModel):
    prediction: Any
    model_version: str


import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from iteratex.model_registry import utils as reg


class DummyModel:
    def predict(self, X):
        return [0] * len(X)


def _load_model():
    try:
        path = reg.load_production_model_path()
        if path is None:
            logger.warning("No production model found; using DummyModel")
            return DummyModel(), "dummy-0"
        
        # Add debug logging
        logger.info(f"Attempting to load model from: {path.absolute()}")
        logger.info(f"File exists: {path.exists()}")
        logger.info(f"File size: {path.stat().st_size if path.exists() else 0} bytes")
        
        # Try loading with .pkl first, then .joblib
        try:
            # First try with the path as is (usually .pkl)
            if path.exists():
                model = joblib.load(path)
            else:
                # Try with .joblib extension if .pkl doesn't exist
                joblib_path = path.with_suffix('.joblib')
                logger.info(f"Trying alternative path: {joblib_path}")
                if joblib_path.exists():
                    model = joblib.load(joblib_path)
                else:
                    raise FileNotFoundError(f"Neither {path} nor {joblib_path} found")
            
            version = path.parent.name
            MODEL_VERSION_GAUGE.labels(version=version).set(1)
            logger.info(f"Successfully loaded production model: {version}")
            return model, version
        except Exception as exc:
            logger.exception(f"Failed to load model from {path}: {str(exc)}")
            return DummyModel(), "dummy-0"
    except Exception as e:
        logger.exception(f"Unexpected error in _load_model: {str(e)}")
        return DummyModel(), "dummy-0"


MODEL, MODEL_VERSION = _load_model()


class _RegistryWatcher(FileSystemEventHandler):
    def on_any_event(self, event):
        global MODEL, MODEL_VERSION
        if event.is_directory:
            return
        # Reload on any change inside registry/production pointer
        MODEL, MODEL_VERSION = _load_model()
        # Update gauge – reset previous labels by clearing and setting new
        MODEL_VERSION_GAUGE.clear()
        MODEL_VERSION_GAUGE.labels(version=MODEL_VERSION).set(1)


def _start_watcher():
    prod_pointer = reg.production_pointer()
    observer = Observer()
    observer.schedule(_RegistryWatcher(), path=str(prod_pointer.parent), recursive=False)
    observer.daemon = True
    observer.start()




@app.on_event("startup")
async def startup_event():
    logger.info("Loaded model version %s", MODEL_VERSION)
    MODEL_VERSION_GAUGE.labels(version=MODEL_VERSION).set(1)
    _start_watcher()


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    with REQUEST_LATENCY.time():
        PREDICTIONS_TOTAL.inc()
        pred = MODEL.predict([req.root])[0]
    return PredictResponse(prediction=pred, model_version=MODEL_VERSION)


@app.get("/health")
async def health():
    return {"status": "ok"}

# Alias for Docker healthcheck
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
