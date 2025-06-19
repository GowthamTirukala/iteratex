"""Shared preprocessing utilities for IteraTex.

Defines a Pydantic model for incoming Kafka records and helper functions to transform
JSON/parquet records into feature matrices and labels to avoid train/serve skew.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.ensemble import IsolationForest


class Record(BaseModel):
    """Minimal schema – adjust fields as needed."""

    feature1: float = Field(...)
    feature2: float = Field(...)
    label: int = Field(..., ge=0, le=1)

    # Accept additional arbitrary keys without validation errors
    class Config:
        extra = "allow"


def validate_json(record: Dict[str, Any]) -> Dict[str, Any]:
    try:
        obj = Record(**record)
        return obj.dict()
    except ValidationError as e:
        raise ValueError(f"Invalid record: {e}") from e


def df_from_parquet(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Ensure required fields exist
    missing = {f for f in ["feature1", "feature2", "label"] if f not in df.columns}
    if missing:
        raise KeyError(f"Missing columns in data: {missing}")
    return df


def reliability_weights(df: pd.DataFrame, contamination: float = 0.02):
    """Return reliability weights (0-1) using IsolationForest anomaly scores.

    Works on *numeric* columns only. If fewer than 2 numeric columns are
    available or IsolationForest fails, falls back to uniform weights (1.0).
    """
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2 or len(num_df) < 10:
        return pd.Series(np.ones(len(df)), index=df.index)

    try:
        iso = IsolationForest(contamination=contamination, random_state=0)
        iso.fit(num_df)
        scores = iso.decision_function(num_df)  # higher = more normal
        weights = 1 / (1 + np.exp(-scores))
        return pd.Series(weights, index=df.index)
    except Exception:
        # Robust fallback – never break training
        return pd.Series(np.ones(len(df)), index=df.index)


def split_features_labels(df: pd.DataFrame, feature_cols: List[str] | None = None):
    """Return X, y where *y* is the ``label`` column and *X* are feature columns.

    Parameters
    ----------
    df
        Dataframe containing at least the label column plus feature columns.
    feature_cols
        Optional explicit list of feature columns (e.g. read from
        ``metadata.json``).  If *None*, all columns except common label names
        will be treated as features.
    """
    label_cols = [c for c in ["label", "target", "y"] if c in df.columns]
    if not label_cols:
        raise KeyError("No label column (label/target/y) found in dataframe")
    if len(label_cols) > 1:
        raise ValueError(f"Multiple label columns found: {label_cols}")
    label_col = label_cols[0]

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns missing from dataframe: {missing}")

    X = df[feature_cols]
    y = df[label_col]
    return X, y
