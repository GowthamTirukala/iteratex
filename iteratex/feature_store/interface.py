"""Thin, synchronous feature-store interface.

*Offline* features are stored as Parquet files under ``data/features/{version}``.
*Online* features are cached in Redis keyed by a SHA-1 of the raw record – this
keeps demo infra simple while mirroring common production setups (Feast, etc.).

Only **minimal** functionality is implemented for now; we can expand later.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import redis  # type: ignore
from loguru import logger

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "features"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REDIS_HOST = os.getenv("ITERATEX_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("ITERATEX_REDIS_PORT", "6379"))
_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client  # noqa: PLW0603
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=False
        )
    return _redis_client


def _hash_record(record: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(record, sort_keys=True).encode()).hexdigest()


class FeatureStore:
    """Simple feature store handling both offline Parquet snapshots and Redis cache."""

    def materialise_snapshot(self, features: pd.DataFrame, version: str) -> Path:
        """Persist *features* dataframe under a given version tag."""
        dest = DATA_DIR / version
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "features.parquet"
        features.to_parquet(path, index=False)
        logger.info("Feature snapshot persisted to {} ({} rows)", path, len(features))
        return path

    # ---------------------------------------------------------------------
    # Online API – key-based put/get
    # ---------------------------------------------------------------------
    def put_online(self, record: Dict[str, Any], features: pd.Series):
        key = _hash_record(record)
        _get_redis().set(key, features.to_json())

    def get_online(self, record: Dict[str, Any]) -> pd.Series | None:
        key = _hash_record(record)
        data = _get_redis().get(key)
        if data is None:
            return None
        return pd.read_json(data, typ="series")

    # ---------------------------------------------------------------------
    # Batch utilities
    # ---------------------------------------------------------------------
    def bulk_put_online(
        self, records: Iterable[Dict[str, Any]], features: pd.DataFrame
    ):
        client = _get_redis()
        with client.pipeline() as pipe:
            for rec, feat in zip(
                records, features.itertuples(index=False), strict=False
            ):
                pipe.set(_hash_record(rec), pd.Series(feat).to_json())
            pipe.execute()

    def bulk_get_online(
        self, records: Iterable[Dict[str, Any]]
    ) -> List[pd.Series | None]:
        client = _get_redis()
        keys = [_hash_record(r) for r in records]
        raw = client.mget(keys)
        return [pd.read_json(x, typ="series") if x else None for x in raw]


__all__ = ["FeatureStore"]
