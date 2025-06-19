"""Read canonical JSON lines and output feature parquet using builder functions.

Usage
-----
python scripts/build_features.py --input data/parsed/parsed.jsonl --output data/features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from iteratex.feature_store import FeatureStore, builders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        in_path = in_path / "parsed.jsonl"
        if not in_path.exists():
            raise FileNotFoundError(
                (
                    "Expected parsed.jsonl inside directory "
                    f"but found none: {in_path.parent}"
                )
            )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines()
    ]

    feats = []
    for rec in records:
        series = pd.concat(
            [
                builders.build_text_features(rec),
                builders.build_meta_features(rec),
            ]
        )
        feats.append(series)

    df = pd.DataFrame(feats)
    fs = FeatureStore()
    fs.materialise_snapshot(df, version="v1")

    logger.info(
        "Built {} feature rows ({} cols) -> {}", len(df), len(df.columns), out_dir
    )


if __name__ == "__main__":
    main()
