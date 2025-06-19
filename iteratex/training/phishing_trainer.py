"""Phishing detection model trainer."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ..model_registry import metadata as reg_meta
from ..preprocessing import reliability_weights


class PhishingTrainer:
    """Train a phishing detection model."""

    def __init__(self, data_path: str, model_dir: str = "models/phishing"):
        """Initialize the trainer.

        Args:
            data_path: Path to the CSV file containing the dataset.
            model_dir: Directory to save the trained model and metadata.
        """
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the phishing dataset."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Basic preprocessing
        # The dataset might have different column names, so let's check for common target column names
        target_col = None
        for col in ["status", "phishing", "is_phishing", "label", "class"]:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            raise ValueError(
                "Could not find target column in the dataset. Expected one of: 'status', 'phishing', 'is_phishing', 'label', 'class'"
            )

        # Convert target to binary (1 for phishing, 0 for legitimate)
        if df[target_col].dtype == "object":
            # If target is string, convert to binary
            df["label"] = (
                df[target_col].str.lower().str.contains("phish|malicious|1")
            ).astype(int)
        else:
            # If target is numeric, assume 1 is phishing
            df["label"] = (df[target_col] > 0).astype(int)

        # Drop the original target column if it's not 'label'
        if target_col != "label":
            df = df.drop(target_col, axis=1)

        # Drop URL column if it exists (we won't use it for training)
        if "url" in df.columns:
            df = df.drop("url", axis=1)

        # Convert any non-numeric columns to numeric if needed
        for col in df.select_dtypes(include=["object"]).columns:
            if col != "label":  # Skip label column
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with missing values
        df = df.dropna()

        # Log class distribution
        class_dist = df["label"].value_counts(normalize=True)
        logger.info(f"Class distribution: {class_dist.to_dict()}")

        return df

    def train(self) -> Dict[str, Any]:
        """Train the model and return metrics."""
        try:
            # Load and preprocess data
            df = self.load_data()

            # Log basic dataset info
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Features: {[col for col in df.columns if col != 'label']}")

            # Split into features and target
            X = df.drop("label", axis=1)
            y = df["label"]

            # Log class distribution
            logger.info(
                f"Class distribution in full dataset: {y.value_counts().to_dict()}"
            )

            # Calculate reliability weights
            logger.info("Calculating reliability weights...")
            weights = reliability_weights(X)

            # Split into train and test sets
            logger.info("Splitting data into train and test sets...")
            X_train, X_test, y_train, y_test, weights_train, _ = train_test_split(
                X, y, weights, test_size=0.2, random_state=42, stratify=y
            )

            logger.info(f"Training set size: {len(X_train)}")
            logger.info(f"Test set size: {len(X_test)}")

            # Train the model
            logger.info("Training model...")
            self.model.fit(X_train, y_train, sample_weight=weights_train)

            # Make predictions
            logger.info("Making predictions...")
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="weighted"),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "class_report": classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                ),
            }

            # Log metrics
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.exception("Full traceback:")
            raise

        # Save the model
        self.save_model(metrics)

        return metrics

    def save_model(self, metrics: Dict[str, Any]) -> None:
        """Save the trained model and metadata."""
        try:
            # Create model directory if it doesn't exist
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # Save the model
            model_path = self.model_dir / "model.joblib"
            joblib.dump(self.model, model_path)

            # Get feature importances if available
            feature_importances = None
            if hasattr(self.model, "feature_importances_"):
                feature_importances = dict(
                    zip(self.model.feature_names_in_, self.model.feature_importances_)
                )

            # Get feature names
            feature_names = (
                list(self.model.feature_names_in_)
                if hasattr(self.model, "feature_names_in_")
                else []
            )

            # Create metadata compatible with ModelRegistry's ModelMetadata
            metadata = reg_meta.ModelMetadata(
                version=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                metrics={
                    "accuracy": float(metrics["accuracy"]),
                    "f1": float(metrics["f1"]),
                    "roc_auc": float(metrics["roc_auc"]),
                },
                features=feature_names,
                hyperparameters={
                    "model_type": "random_forest",
                    "n_estimators": (
                        self.model.n_estimators
                        if hasattr(self.model, "n_estimators")
                        else 100
                    ),
                    "max_depth": (
                        self.model.max_depth if hasattr(self.model, "max_depth") else 10
                    ),
                },
                training_data_version="phishing_dataset_v1",  # In production, use a hash of the dataset
            )

            # Save feature importances separately if available
            if feature_importances and hasattr(self.model, "feature_importances_"):
                importances_path = self.model_dir / "feature_importances.json"
                with open(importances_path, "w") as f:
                    json.dump(
                        {
                            "feature_importances": feature_importances,
                            "features": feature_names,
                            "model_version": metadata.version,
                        },
                        f,
                        indent=2,
                    )

            # Save metadata
            metadata_path = self.model_dir / "metadata.json"
            metadata.save_json(metadata_path)

            # Save feature importances to a CSV file for easier analysis
            if feature_importances:
                importances_path = self.model_dir / "feature_importances.csv"
                pd.DataFrame(
                    {
                        "feature": feature_importances.keys(),
                        "importance": feature_importances.values(),
                    }
                ).sort_values("importance", ascending=False).to_csv(
                    importances_path, index=False
                )

            logger.info("\n=== Model Training Complete ===")
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Metadata saved to: {metadata_path}")

            if feature_importances:
                logger.info(f"Feature importances saved to: {importances_path}")

            return metadata

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.exception("Full traceback:")
            raise


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print model metrics in a formatted way."""
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    # Print classification report
    print("\n=== Classification Report ===")
    class_report = metrics["class_report"]

    # Print per-class metrics
    print("\nPer-class metrics:")
    rows = []
    for class_name, metrics in class_report.items():
        if isinstance(
            metrics, dict
        ):  # Skip non-class entries like 'accuracy', 'macro avg', etc.
            rows.append(
                [
                    class_name,
                    f"{metrics.get('precision', 0):.4f}",
                    f"{metrics.get('recall', 0):.4f}",
                    f"{metrics.get('f1-score', 0):.4f}",
                    f"{metrics.get('support', 0):,}",
                ]
            )

    # Print the table
    from tabulate import tabulate

    print(
        tabulate(
            rows,
            headers=["Class", "Precision", "Recall", "F1-Score", "Support"],
            tablefmt="grid",
        )
    )

    # Print overall metrics
    print("\nOverall metrics:")
    print(f"Accuracy: {class_report.get('accuracy', 0):.4f}")
    if "macro avg" in class_report:
        print(f"Macro F1: {class_report['macro avg']['f1-score']:.4f}")
    if "weighted avg" in class_report:
        print(f"Weighted F1: {class_report['weighted avg']['f1-score']:.4f}")


def main():
    """Train the phishing detection model."""
    try:
        # Path to the dataset
        data_path = r"C:\Users\tiruk\Downloads\Web page Phishing Detection Dataset\dataset_phishing.csv"

        print("=== Phishing Detection Model Training ===")
        print(f"Dataset: {data_path}")

        # Initialize and train the model
        trainer = PhishingTrainer(data_path)
        metrics = trainer.train()

        # Save the model and get metadata
        metadata = trainer.save_model(metrics)

        # Print metrics
        print_metrics(metrics)

        print("\nTraining completed successfully!")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
