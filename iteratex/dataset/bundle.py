"""Utilities to merge labeled & pseudo-labeled datasets into a single training parquet.

The function `bundle_datasets` loads two parquet snapshots (typically tracked
by DVC), merges them, removes duplicates, computes *reliability weights* using
`iteratex.preprocessing.reliability_weights`, and writes the unified dataframe
under `data/training/{version}.parquet`.

A small CLI stub is provided so the script can be invoked directly or as a
DVC stage::

    python -m iteratex.dataset.bundle --labeled data/labeled/v1.parquet \
        --pseudo data/pseudo/v1.parquet --version v1
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from iteratex.preprocessing import df_from_parquet, reliability_weights

# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def bundle_datasets(
    labeled_path: Path | str,
    pseudo_path: Path | str,
    output_root: Path | str = "data/training",
    version: Optional[str] = None,
) -> Path:
    """Create unified training dataset.

    Parameters
    ----------
    labeled_path : Path or str
        Parquet file containing *human-labeled* examples.
    pseudo_path : Path or str
        Parquet file containing pseudo-labeled examples (e.g. self-training output).
    output_root : Path or str, default "data/training"
        Directory where bundled parquet will be stored.
    version : str, optional
        Version / snapshot tag.  If *None* a hash of input paths + timestamp is used.

    Returns
    -------
    Path
        Path to the written parquet snapshot.
    """
    labeled_path = Path(labeled_path)
    pseudo_path = Path(pseudo_path)
    output_root = Path(output_root)

    logger.info("Loading labeled data from {}", labeled_path)
    df_labeled = df_from_parquet(labeled_path)

    logger.info("Loading pseudo-labeled data from {}", pseudo_path)
    df_pseudo = df_from_parquet(pseudo_path)

    logger.info("Merging datasets – labeled: {}, pseudo: {}", len(df_labeled), len(df_pseudo))
    df = pd.concat([df_labeled, df_pseudo], ignore_index=True)

    # Deduplicate: assume an 'id' column if present, else drop duplicate rows.
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="id")
        logger.info("Removed {} duplicate ids", before - len(df))
    else:
        before = len(df)
        df = df.drop_duplicates()
        logger.info("Removed {} fully-duplicate rows", before - len(df))

    # Add / recompute reliability weights
    df["weight"] = reliability_weights(df)

    # Determine version tag
    if version is None:
        tag_src = f"{labeled_path}:{pseudo_path}:{datetime.utcnow().isoformat()}"
        version = hashlib.sha1(tag_src.encode()).hexdigest()[:8]

    out_dir = output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{version}.parquet"

    df.to_parquet(out_path, index=False)
    logger.success("Bundled dataset saved to {} ({} rows, {} cols)", out_path, len(df), len(df.columns))
    return out_path


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli():  # pragma: no cover – simple wrapper
    parser = argparse.ArgumentParser(description="Bundle labeled & pseudo-labeled datasets")
    parser.add_argument("--labeled", required=True, help="Path to labeled parquet snapshot")
    parser.add_argument("--pseudo", required=True, help="Path to pseudo-labeled parquet snapshot")
    parser.add_argument("--output-root", default="data/training", help="Root directory for output")
    parser.add_argument("--version", default=None, help="Snapshot version tag (optional)")
    args = parser.parse_args()

    bundle_datasets(args.labeled, args.pseudo, args.output_root, args.version)


if __name__ == "__main__":
    _cli()
