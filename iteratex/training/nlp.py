"""Simple NLP text classification trainer using scikit-learn's SGDClassifier."""

from pathlib import Path
from typing import Any, Dict

import joblib
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from iteratex.preprocessing import df_from_parquet, reliability_weights

from .base import BaseTrainer


class SimpleTextClassifierTrainer(BaseTrainer):
    def suggest_params(self, trial):  # noqa: D401
        return {
            "max_features": trial.suggest_int("max_features", 1000, 10000, step=1000),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        }

    def train(
        self,
        data_path: Path,
        output_dir: Path,
        **hyperparams,
    ) -> Dict[str, Any]:
        df = df_from_parquet(data_path)
        if "text" not in df.columns:
            raise KeyError("Column 'text' not present for NLP training")
        X_raw = df["text"]
        y = df["label"]

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, random_state=42
        )

        max_features = hyperparams.get("max_features", 5000)
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)

        alpha = hyperparams.get("alpha", 1e-4)
        clf = SGDClassifier(loss="log_loss", alpha=alpha)
        weights = reliability_weights(df)
        clf.fit(X_train, y_train, sample_weight=weights.loc[X_train.index])
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        metrics = {"accuracy": acc}

        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": vectorizer, "model": clf}, output_dir / "model.pkl")
        logger.info("Trained simple text classifier with accuracy %.4f", acc)
        return metrics
