"""Utility functions to *detect* payload type and *parse* it into a canonical
Python dictionary so that downstream feature builders can work on a uniform
schema.

Although the heuristics below are simple they cover 90 % of real-world ingress
cases (JSON, plain text, small CSV blobs). They can be improved later (e.g.
MIME sniffing, magic numbers, configurable CSV dialects).
"""

from __future__ import annotations

import csv
import json
import logging
from enum import Enum, auto
from io import StringIO
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class RecordType(Enum):
    """Supported raw payload formats."""

    JSON = auto()
    CSV = auto()
    TEXT = auto()


def detect_type(raw: Union[str, bytes]) -> RecordType:
    """Best-effort detection of *raw* payload format.

    Parameters
    ----------
    raw
        Message body as ``str`` or ``bytes`` coming from the message broker / API.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    sample = raw.lstrip()[:100]
    if sample.startswith("{") or sample.startswith("["):
        return RecordType.JSON
    if "," in sample and "\n" in sample:
        return RecordType.CSV
    return RecordType.TEXT


def parse(raw: Union[str, bytes]) -> Dict[str, Any]:
    """Parse *raw* payload into a canonical dict.

    For TEXT payloads the result has a single key ``body`` containing the text.
    For CSV payloads the first row is treated as the header and only the first
    record is returned (real pipelines should expand this). JSON is loaded as-is.
    """
    rtype = detect_type(raw)
    if rtype is RecordType.JSON:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON payload", exc_info=exc)
            raise

    if rtype is RecordType.CSV:
        text = raw.decode() if isinstance(raw, bytes) else raw
        reader = csv.DictReader(StringIO(text))
        try:
            first_row = next(reader)
            return dict(first_row)
        except StopIteration:
            raise ValueError("CSV payload contained no rows") from None

    # TEXT fallback
    text = raw.decode() if isinstance(raw, bytes) else raw
    return {"body": text}


__all__: List[str] = [
    "RecordType",
    "detect_type",
    "parse",
]
