"""Evaluator that compares new run metrics against production and decides promotion."""

import json
from typing import Dict

from loguru import logger

from iteratex.model_registry import utils as reg


class Evaluator:
    def __init__(self, primary_metric: str = "accuracy", higher_is_better: bool = True):
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better

    def _metric_value(self, metrics: Dict[str, float]) -> float:
        if self.primary_metric not in metrics:
            raise KeyError(f"Metric {self.primary_metric} not present in metrics.json")
        return float(metrics[self.primary_metric])

    def should_promote(self, candidate_metrics: Dict[str, float]) -> bool:
        prod_run = reg.current_production_run()
        if prod_run is None:
            logger.info("No production model yet – will promote candidate by default")
            return True

        prod_metrics_path = reg.runs_dir() / prod_run / "metrics.json"
        if not prod_metrics_path.exists():
            logger.warning("Production metrics missing; promoting candidate")
            return True
        with prod_metrics_path.open() as f:
            prod_metrics = json.load(f)

        cand_val = self._metric_value(candidate_metrics)
        prod_val = self._metric_value(prod_metrics)
        better = cand_val > prod_val if self.higher_is_better else cand_val < prod_val
        logger.info(
            "Candidate %s %.4f vs production %.4f => better=%s",
            self.primary_metric,
            cand_val,
            prod_val,
            better,
        )
        return better
