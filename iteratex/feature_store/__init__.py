"""Feature store package.

This sub-package exposes *builder* utilities that know how to generate feature
matrices from canonical records **and** a thin interface for persisting /
retrieving those feature vectors both offline and online.

Design goals:
1. Avoid train/serve skew – the very same builder functions are imported by the
   FastAPI inference service.
2. Keep external storage pluggable.  We start with Parquet (offline) and Redis
   (online) because they do not require additional infra locally.
"""

from __future__ import annotations

from .builders import build_meta_features, build_text_features  # noqa: F401
from .interface import FeatureStore  # noqa: F401

__all__ = [
    "build_text_features",
    "build_meta_features",
    "FeatureStore",
]
