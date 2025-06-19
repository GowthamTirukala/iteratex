"""CLI script to train, evaluate, and (optionally) promote a new model run.

Usage:
    python scripts/retrain.py --trainer=dummy --data=data/training/training_data.parquet
"""
import argparse
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime

from loguru import logger

from iteratex.training.base import DummyClassifierTrainer
from iteratex.training.regression import DummyRegressionTrainer
from iteratex.training.nlp import SimpleTextClassifierTrainer
from iteratex.evaluation.evaluator import Evaluator
from iteratex.model_registry.metadata import ModelMetadata
from iteratex.model_registry import utils as reg


TRAINERS = {
    "dummy": DummyClassifierTrainer,
    "regression_dummy": DummyRegressionTrainer,
    "text_classifier": SimpleTextClassifierTrainer,
}


def main():
    parser = argparse.ArgumentParser(description="IteraTex retraining job")
    parser.add_argument("--trainer", default="dummy", choices=TRAINERS.keys())
    parser.add_argument("--metric", default="accuracy", help="Primary metric for evaluator (accuracy, rmse, etc.)")
    parser.add_argument("--data", required=True, help="Path to training data (Parquet)")
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = reg.create_run_dir(run_id)

    trainer_cls = TRAINERS[args.trainer]
    trainer = trainer_cls()

    data_path = Path(args.data)
    metrics = trainer.train(data_path=data_path, output_dir=run_dir)

    # ------------------------------------------------------------------
    # Persist run metadata (lineage & reproducibility)
    # ------------------------------------------------------------------
    model_path = run_dir / "model.pkl"
    try:
        import joblib

        model = joblib.load(model_path)
        hyperparams = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    except Exception:
        logger.exception("Failed to load model for metadata – storing empty hyperparameters")
        hyperparams = {}

    # Compute simple hash of training data for lineage
    h = hashlib.sha256()
    with open(data_path, "rb") as _f:
        while chunk := _f.read(8192):
            h.update(chunk)
    data_hash = h.hexdigest()

    # Infer feature list from data columns (exclude common label column names)
    try:
        from iteratex.preprocessing import df_from_parquet

        df_cols = df_from_parquet(data_path).columns
        features = [c for c in df_cols if c not in {"label", "target", "y"}]
    except Exception:
        logger.exception("Could not infer feature list from data; leaving empty")
        features = []

    meta = ModelMetadata(
        version=run_id,
        metrics=metrics,
        created_at=datetime.utcnow(),
        features=features,
        hyperparameters=hyperparams,
        training_data_version=data_hash,
    )
    meta.save_json(run_dir / "metadata.json")

    # Keep legacy metrics.json for backward compatibility
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    # Evaluate & promote
    higher_is_better = False if args.metric.lower() == "rmse" else True
    evaluator = Evaluator(primary_metric=args.metric, higher_is_better=higher_is_better)
    if evaluator.should_promote(metrics):
        reg.promote(run_id)
    else:
        logger.info("Candidate not better – keeping production unchanged")


if __name__ == "__main__":
    main()
