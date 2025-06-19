"""Simple smoke test for Optuna HPO utility."""

from pathlib import Path

import pandas as pd

from iteratex.training.optuna_utils import run_study
from iteratex.training.regression import DummyRegressionTrainer


def test_optuna_run(tmp_path: Path):
    # Create tiny regression dataset
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "y": [1.0, 1.9, 3.1, 3.9]})
    data_path = tmp_path / "reg.parquet"
    df.to_parquet(data_path, index=False)

    best = run_study(
        trainer_factory=DummyRegressionTrainer,
        data_path=data_path,
        metric="rmse",
        direction="minimize",
        n_trials=2,
        study_name="test",
    )
    # Should return strategy param
    assert best["strategy"] in {"mean", "median"}
