"""Simple regression trainer using DummyRegressor and RMSE metric."""
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

from iteratex.preprocessing import df_from_parquet, split_features_labels, reliability_weights
from iteratex.model_registry import utils as reg
from .base import BaseTrainer


class DummyRegressionTrainer(BaseTrainer):
    """Baseline regression trainer (predicts mean)."""

    def train(self, data_path: Path, output_dir: Path) -> Dict[str, Any]:
        df = df_from_parquet(data_path)
        prod_run = reg.current_production_run()
        feature_cols = None
        if prod_run:
            meta = reg.load_metadata(prod_run)
            if meta and meta.features:
                feature_cols = meta.features
        X, y = split_features_labels(df, feature_cols)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = DummyRegressor(strategy="mean")
        weights = reliability_weights(df)
        reg.fit(X_train, y_train, sample_weight=weights.loc[X_train.index])
        preds = reg.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics = {"rmse": rmse}

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pkl"
        joblib.dump(reg, model_path)
        logger.info("Trained dummy regressor with RMSE %.4f, saved to %s", rmse, model_path)

        return metrics
