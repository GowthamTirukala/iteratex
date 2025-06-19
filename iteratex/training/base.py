"""Generic trainer base class and a simple trainer implementation (dummy)."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib


class BaseTrainer(ABC):
    """Common interface all trainers must implement."""

    @abstractmethod
    def train(self, data_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Train a model.

        Returns metrics dict.
        """


class DummyClassifierTrainer(BaseTrainer):
    """Simple trainer for demo purposes."""

    def train(self, data_path: Path, output_dir: Path) -> Dict[str, Any]:
        from iteratex.preprocessing import df_from_parquet, split_features_labels, reliability_weights
        from iteratex.model_registry import utils as reg
        df = df_from_parquet(data_path)
        prod_run = reg.current_production_run()
        feature_cols = None
        if prod_run:
            meta = reg.load_metadata(prod_run)
            if meta and meta.features:
                feature_cols = meta.features
        X, y = split_features_labels(df, feature_cols)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        weights = reliability_weights(df)
        acc = accuracy_score(y_test, preds, sample_weight=weights.loc[y_test.index])
        metrics = {"accuracy": acc}

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pkl"
        joblib.dump(clf, model_path)
        logger.info("Trained dummy classifier with accuracy %.4f, saved to %s", acc, model_path)

        return metrics
