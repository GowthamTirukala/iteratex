"""Parse raw data files in *input* directory into canonical JSONL rows.

Usage
-----
python scripts/parse_raw.py --input data/raw --output data/parsed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from loguru import logger

from iteratex.preprocessing import parser as p


def _parse_file(path: Path) -> List[dict]:
    raw = path.read_bytes()
    rec = p.parse(raw)
    return [rec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder containing raw files")
    ap.add_argument(
        "--output", required=True, help="Destination folder for jsonl files"
    )
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "parsed.jsonl"

    n_processed = 0
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for file in in_dir.iterdir():
            if file.is_dir():
                continue
            try:
                records = _parse_file(file)
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
                    n_processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse {}: {}", file, exc)

    logger.info(
        "Parsed {} files → {} records -> {}",
        len(list(in_dir.iterdir())),
        n_processed,
        jsonl_path,
    )


if __name__ == "__main__":
    main()
