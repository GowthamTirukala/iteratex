"""Unit tests for dataset bundling utility."""

from pathlib import Path

import pandas as pd

from iteratex.dataset import bundle_datasets


def test_bundle_datasets(tmp_path: Path):
    # create tiny labeled & pseudo datasets
    labeled = pd.DataFrame(
        {
            "id": [1, 2],
            "feature": [10, 20],
            "label": [0, 1],
        }
    )
    pseudo = pd.DataFrame(
        {
            "id": [2, 3],
            "feature": [20, 30],
            "label": [1, 0],
        }
    )

    labeled_path = tmp_path / "labeled.parquet"
    pseudo_path = tmp_path / "pseudo.parquet"
    labeled.to_parquet(labeled_path, index=False)
    pseudo.to_parquet(pseudo_path, index=False)

    out_path = bundle_datasets(
        labeled_path=labeled_path,
        pseudo_path=pseudo_path,
        output_root=tmp_path,
        version="test",
    )

    bundled = pd.read_parquet(out_path)
    # Expect 3 unique ids after deduplication
    assert len(bundled) == 3
    # Weight column should exist
    assert "weight" in bundled.columns
