from __future__ import annotations

"""Lightweight helper wrappers around MLflow so the rest of the codebase
can import a single utility module instead of sprinkling MLflow calls
throughout the code.  These wrappers purposely cover only the pieces we
need for Phase 1 (local file backend, simple experiment retrieval, and
common logging helpers)."""

from pathlib import Path
from typing import Any, Dict

import mlflow
from loguru import logger

# -----------------------------------------------------------------------------
# Experiment helpers
# -----------------------------------------------------------------------------

def get_or_create_experiment(name: str = "IteraTex") -> str:
    """Return the experiment ID for *name*, creating it if necessary."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id  # type: ignore[return-value]
    return mlflow.create_experiment(name)


# -----------------------------------------------------------------------------
# Logging wrappers
# -----------------------------------------------------------------------------

def log_params_flat(params: Dict[str, Any]) -> None:
    """Flatten nested params into a single-level dict and log via mlflow."""
    flat: Dict[str, Any] = {}

    def _flatten(prefix: str, obj: Dict[str, Any]):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(key, v)
            else:
                flat[key] = v

    _flatten("", params)
    mlflow.log_params(flat)


def log_metrics(metrics: Dict[str, float]) -> None:
    """Log metric key/value pairs regardless of numeric type handling."""
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            logger.warning("Failed to log metric %s=%s", k, v)


# -----------------------------------------------------------------------------
# Tracking URI convenience
# -----------------------------------------------------------------------------


def configure_local_tracking(base_dir: Path | str = "mlruns") -> None:
    """Point MLflow tracking URI to a local folder (default 'mlruns/')."""
    d = Path(base_dir).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{d}")


__all__ = [
    "configure_local_tracking",
    "get_or_create_experiment",
    "log_metrics",
    "log_params_flat",
]
