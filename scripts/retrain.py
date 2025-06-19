"""CLI script to train, evaluate, and (optionally) promote a new model run.

Usage:
    python scripts/retrain.py --trainer=dummy --data=data/training/training_data.parquet
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# ---- MLflow helpers ---------------------------------------------------------
import mlflow
from loguru import logger

from iteratex.evaluation.evaluator import Evaluator
from iteratex.model_registry import utils as reg
from iteratex.model_registry.metadata import ModelMetadata
from iteratex.training.base import DummyClassifierTrainer
from iteratex.training.nlp import SimpleTextClassifierTrainer
from iteratex.training.regression import DummyRegressionTrainer
from iteratex.utils import mlflow_utils as mlfu

TRAINERS = {
    "dummy": DummyClassifierTrainer,
    "regression_dummy": DummyRegressionTrainer,
    "text_classifier": SimpleTextClassifierTrainer,
}


def main():
    parser = argparse.ArgumentParser(description="IteraTex retraining job")
    parser.add_argument("--trainer", default="dummy", choices=TRAINERS.keys())
    parser.add_argument(
        "--metric",
        default="accuracy",
        help="Primary metric for evaluator (accuracy, rmse, etc.)",
    )
    parser.add_argument("--data", required=True, help="Path to training data (Parquet)")
    parser.add_argument(
        "--hpo", action="store_true", help="Enable Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of Optuna trials when --hpo"
    )
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------------
    # MLflow setup – local file backend, create experiment if missing
    # ------------------------------------------------------------------
    mlfu.configure_local_tracking()
    # Enable automatic logging of sklearn params, metrics, artifacts
    mlflow.autolog(log_models=True, disable=False)
    experiment_id = mlfu.get_or_create_experiment()
    mlflow.start_run(experiment_id=experiment_id, run_name=run_id)
    mlflow.set_tag("trainer", args.trainer)
    run_dir = reg.create_run_dir(run_id)

    best_params: Dict[str, Any] = {}

    trainer_cls = TRAINERS[args.trainer]
    trainer = trainer_cls()  # type: ignore[abstract]

    data_path = Path(args.data)

    if args.hpo:
        from iteratex.training.optuna_utils import run_study

        best_params = run_study(
            trainer_factory=trainer_cls,
            data_path=data_path,
            metric=args.metric,
            direction=("maximize" if args.metric in {"accuracy"} else "minimize"),
            n_trials=args.n_trials,
            study_name=f"{args.trainer}-{run_id}",
        )
        mlfu.log_params_flat(best_params)
        trainer = trainer_cls()  # type: ignore[abstract]  # fresh instance
        metrics = trainer.train(data_path=data_path, output_dir=run_dir, **best_params)
    else:
        metrics = trainer.train(data_path=data_path, output_dir=run_dir)

    # Log primary metrics to MLflow ASAP so we see progress even if later steps fail
    mlfu.log_metrics(metrics)

    # ------------------------------------------------------------------
    # Persist run metadata (lineage & reproducibility)
    # ------------------------------------------------------------------
    model_path = run_dir / "model.pkl"
    try:
        import joblib

        model = joblib.load(model_path)
        hyperparams = (
            model.get_params(deep=False) if hasattr(model, "get_params") else {}
        )
        # Log hyperparameters
        mlfu.log_params_flat(hyperparams)
    except Exception:
        logger.exception(
            "Failed to load model for metadata – storing empty hyperparameters"
        )
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

        # Log artefacts to MLflow (model + metadata soon-to-be-written)
    mlflow.log_artifact(str(model_path), artifact_path="model")

    # Choose hyperparameters source: HPO best or model.get_params fallback
    metadata_hparams = best_params if best_params else hyperparams

    meta = ModelMetadata(
        version=run_id,
        metrics=metrics,
        created_at=datetime.utcnow(),
        features=features,
        hyperparameters=metadata_hparams,
        training_data_version=data_hash,
        mlflow_run_id=mlflow.active_run().info.run_id,
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

    # Close MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()
