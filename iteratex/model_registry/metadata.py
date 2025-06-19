"""Model metadata representation and helpers.

This module defines ``ModelMetadata`` – a pydantic model that captures the most
important lineage information about a trained model artefact.  The metadata is
stored next to the artefacts inside ``registry/runs/<run_id>/`` so that every
run directory is self-describing.

Fields
------
version
    Unique identifier of the run / model version (e.g. timestamp).
metrics
    Primary and auxiliary evaluation metrics collected during training.
created_at
    UTC ISO-8601 timestamp when the model was created.
features
    Ordered list of feature names used when fitting the model.
hyperparameters
    Mapping of hyper-parameter name → value that fully specifies the training
    configuration (e.g. result of ``estimator.get_params()`` for scikit-learn
    models).
training_data_version
    Content hash (SHA-256) or dataset ID that the model was trained on.  This
    links the model artefact back to the exact training data snapshot.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, validator


class ModelMetadata(BaseModel):
    version: str = Field(..., description="Model / run identifier")
    metrics: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    features: List[str]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_data_version: str = Field(
        ..., description="Hash / ID of training data snapshot"
    )
    mlflow_run_id: str | None = Field(
        default=None, description="Linked MLflow run identifier"
    )

    class Config:
        extra = "forbid"

    @validator("created_at", pre=True)
    def _ensure_dt(cls, v):  # noqa: D401
        """Ensure ``created_at`` is an aware UTC datetime instance."""
        if isinstance(v, datetime):
            return v
        # Assume ISO string
        return datetime.fromisoformat(v)

    # ----- Persistence helpers -------------------------------------------------

    def save_json(self, path: Path) -> None:
        """Write metadata to *path* as pretty-formatted JSON."""
        # Pydantic v2: use `model_dump_json` instead of deprecated `json` kwargs
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: Path) -> "ModelMetadata":
        """Load metadata from a JSON file."""
        # Pydantic v2: parse JSON string via `model_validate_json`
        return cls.model_validate_json(path.read_text())


# Convenience wrappers so callers do not need to import pydantic directly
__all__ = ["ModelMetadata"]  # noqa: E305
