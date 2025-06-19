"""Script to register a trained model in the model registry."""

import shutil
from datetime import datetime
from pathlib import Path


def register_model(source_dir: str, run_id: str = None):
    """Register a trained model in the model registry.

    Args:
        source_dir: Directory containing the trained model and metadata
        run_id: Optional run ID. If not provided, uses current timestamp
    """
    # Define paths
    base_dir = Path(__file__).parent.parent
    source_path = (
        Path(source_dir) if Path(source_dir).is_absolute() else base_dir / source_dir
    )

    # Generate run ID if not provided
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Set up registry paths
    registry_root = base_dir / "registry"
    runs_dir = registry_root / "runs"
    run_path = runs_dir / run_id
    prod_pointer = registry_root / "production"

    # Create necessary directories
    run_path.mkdir(parents=True, exist_ok=True)

    # Copy model and metadata
    shutil.copy2(source_path / "model.joblib", run_path / "model.pkl")
    shutil.copy2(source_path / "metadata.json", run_path / "metadata.json")

    # Copy feature importance files if they exist
    for file in ["feature_importances.csv", "feature_importances.json"]:
        src = source_path / file
        if src.exists():
            shutil.copy2(src, run_path / file)

    # Update production pointer
    with open(prod_pointer, "w") as f:
        f.write(f"runs/{run_id}")

    print(f"✅ Model registered in {run_path}")
    print(f"🎯 Production pointer set to {run_id}")
    print(
        "\nModel registered successfully! The API will now use this model for predictions."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Register a trained model in the model registry."
    )
    parser.add_argument(
        "--source-dir",
        default="models/phishing",
        help="Directory containing the trained model and metadata",
    )
    parser.add_argument(
        "--run-id", help="Optional run ID. If not provided, uses current timestamp"
    )

    args = parser.parse_args()
    register_model(args.source_dir, args.run_id)
