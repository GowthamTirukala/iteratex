"""Utility helpers to run Optuna hyper-parameter optimisation for IteraTex trainers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict

import optuna
from loguru import logger

from iteratex.training.base import BaseTrainer


def run_study(
    trainer_factory: Callable[[], BaseTrainer],
    data_path: Path,
    metric: str,
    direction: str = "minimize",
    n_trials: int = 20,
    study_name: str | None = None,
    storage: str | None = None,
) -> Dict[str, Any]:
    """Run Optuna study and return best hyper-parameters.

    Parameters
    ----------
    trainer_factory
        Callable returning fresh trainer instance (no side-effects between trials).
    data_path
        Training dataset path.
    metric
        Metric key to optimise (must be in metrics dict returned by trainer.train).
    direction
        "minimize" or "maximize".
    n_trials
        Number of trials.
    study_name
        Optional study name.
    storage
        Optional Optuna storage URI (e.g. sqlite:///optuna.db) for persistence.
    """

    study = optuna.create_study(
        direction=direction, study_name=study_name, storage=storage
    )

    def _objective(trial: optuna.Trial):  # noqa: D401
        trainer = trainer_factory()
        # Ask trainer for suggested params (may be empty dict)
        params = trainer.suggest_params(trial)
        with TemporaryDirectory() as tmpdir:
            metrics = trainer.train(data_path, Path(tmpdir), **params)
        if metric not in metrics:
            raise KeyError(f"Metric '{metric}' not found in trainer metrics: {metrics}")
        value = metrics[metric]
        logger.debug("Trial {} params={} value={}", trial.number, params, value)
        return value

    logger.info("Starting Optuna study ({} trials, direction={})", n_trials, direction)
    study.optimize(_objective, n_trials=n_trials)
    logger.success(
        "HPO finished – best value {:.4f}, params {}",
        study.best_value,
        study.best_params,
    )
    return study.best_params
