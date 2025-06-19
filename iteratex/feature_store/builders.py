"""Functions that convert *canonical* record dicts into engineered feature
columns.

These builders return a **Pandas Series** so that the caller can easily combine
multiple builder outputs using ``pd.concat`` – this keeps the interface simple
and avoids assumptions about downstream frameworks (spark / pandas / polars).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# A small singleton TF-IDF model that is lazy-trained on demand.  In real
# production you would persist this vectoriser and load it with the model.
_vectoriser: TfidfVectorizer | None = None


def _get_vectoriser() -> TfidfVectorizer:
    global _vectoriser  # noqa: PLW0603
    if _vectoriser is None:
        _vectoriser = TfidfVectorizer(max_features=512, stop_words="english")
    return _vectoriser


def build_text_features(record: Dict[str, Any]) -> pd.Series:
    """Return TF-IDF vector (dense) + simple text statistics."""
    body: str = str(record.get("body") or record.get("text") or "")
    vec = _get_vectoriser().fit_transform([body])  # (1, n_features) sparse
    dense = vec.toarray().ravel()

    stats = pd.Series(
        {
            "len_chars": len(body),
            "num_digits": len(re.findall(r"\d", body)),
            "num_links": len(re.findall(r"https?://", body)),
        }
    )
    tfidf_series = pd.Series(dense, index=[f"tfidf_{i}" for i in range(len(dense))])
    return pd.concat([stats, tfidf_series])


def build_meta_features(record: Dict[str, Any]) -> pd.Series:
    """Example of shallow metadata features that are format-agnostic."""
    return pd.Series({"has_label": int("label" in record), "num_fields": len(record)})


__all__: List[str] = [
    "build_text_features",
    "build_meta_features",
]
