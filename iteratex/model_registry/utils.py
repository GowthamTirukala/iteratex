"""Helper functions for IteraTex model registry.

Folder structure (root of repo):
registry/
    runs/<run_id>/
        model.pkl
        metrics.json
    production -> runs/<run_id>/  # symbolic link (or text file path on Windows)

This module is **code**, not the artifact directory. It is safe even if `.gitignore` excludes the artifact folder.
"""
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .metadata import ModelMetadata

from loguru import logger

# --- Project root detection ----------------------------------------------------
# By default we assume the code lives in an editable checkout and the registry
# directory is located at the repo root.  When iteratex is installed into the
# global site-packages and used outside the repo (e.g. notebook in another
# location), we want to fall back to either an explicit environment variable or
# walk upwards from the current working directory to find a folder that
# contains a `registry` subdir.  This makes promotion scripts location-agnostic
# and prevents the "two separate registries" problem we hit earlier.


def _detect_project_root() -> Path:
    """Return the IteraTex project root directory.

    Priority:
    1. Explicit ``ITERATEX_PROJECT_ROOT`` environment variable.
    2. First parent of ``cwd`` that contains a ``registry`` folder.
    3. Fallback: two levels up from this utils.py (installed package).
    """
    env = os.getenv("ITERATEX_PROJECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "registry").exists():
            return parent
    # fallback – site-packages install path
    return Path(__file__).resolve().parents[2]


ROOT_DIR = _detect_project_root()


def _registry_root() -> Path:
    return ROOT_DIR / "registry"


def runs_dir() -> Path:
    d = _registry_root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def production_pointer() -> Path:
    return _registry_root() / "production"


def create_run_dir(run_id: str) -> Path:
    path = runs_dir() / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_metrics(metrics_path: Path) -> Dict[str, float]:
    with metrics_path.open() as f:
        return json.load(f)


def current_production_run() -> Optional[str]:
    prod = production_pointer()
    if not prod.exists():
        return None

    # If symbolic link (UNIX) – read target name
    if prod.is_symlink():
        target = os.readlink(prod)
        return Path(target).name

    # If text file (Windows fallback)
    try:
        content = prod.read_text().strip()
        return Path(content).name if content else None
    except Exception:
        return None


def _write_pointer_atomic(run_path: Path):
    """Atomically update the *production* pointer to *run_path*.

    A temporary file/symlink is written first and then **renamed** to the final
    location in a single operation to avoid readers seeing a half-written
    pointer.
    """
    prod = production_pointer()
    tmp = prod.with_suffix('.tmp')

    # Cleanup any stale temp file
    if tmp.exists():
        tmp.unlink()

    # Remove existing production pointer *after* tmp cleaned so we always keep
    # at least one valid pointer on disk.
    if prod.exists():
        prod.unlink()

    rel_path = Path("runs") / run_path.name
    try:
        # Prefer relative symlink when possible – portable inside Docker volume
        tmp.symlink_to(rel_path)
    except (OSError, NotImplementedError):
        # Symlink not permitted → write relative path text file (works everywhere)
        tmp.write_text(str(rel_path))

    # Atomic rename – on POSIX this is guaranteed, on Windows it's best-effort
    tmp.replace(prod)


import joblib
import numpy as np
from iteratex.preprocessing import Record


def _smoke_test(run_path: Path) -> None:
    """Load the candidate model and run a minimal prediction to ensure it is usable.

    This prevents a broken model from being promoted and taking down the API.
    """
    model_file = run_path / "model.pkl"
    model = joblib.load(model_file)

    # Use feature schema from metadata if available, else default two-feature stub
    meta_path = run_path / "metadata.json"
    if meta_path.exists():
        from .metadata import ModelMetadata
        m = ModelMetadata.load_json(meta_path)
        # Use string for URL-like feature, 0 for others
        sample = {f: ("test.com" if f.lower() == "url" else 0) for f in m.features[:100]}
    else:
        sample = {"feature1": 0, "feature2": 0}

    import pandas as pd

    try:
        # Prefer DataFrame input (covers most sklearn pipelines with ColumnTransformer)
        _ = model.predict(pd.DataFrame([sample]))
    except Exception:
        try:
            # Fallback to list-of-dict for models trained that way
            _ = model.predict([sample])
        except Exception as exc:
            raise RuntimeError(f"Smoke test failed: {exc}") from exc


def promote(run_id: str):
    run_path = runs_dir() / run_id
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    # Basic validation – ensure artefacts exist
    for fname in ("model.pkl", "metadata.json"):
        if not (run_path / fname).exists():
            raise FileNotFoundError(run_path / fname)

    # Final sanity: run smoke test before switching pointer
    _smoke_test(run_path)

    _write_pointer_atomic(run_path)
    

    logger.info("Promoted run %s to production", run_id)



# ---------------- Convenience helpers ----------------------------------------

def list_runs() -> List[str]:
    """Return available run ids sorted newest→oldest."""
    return sorted((p.name for p in runs_dir().iterdir() if p.is_dir()), reverse=True)


def load_metadata(run_id: str) -> Optional[ModelMetadata]:
    """Load ``metadata.json`` for *run_id* if present."""
    path = runs_dir() / run_id / "metadata.json"
    return ModelMetadata.load_json(path) if path.exists() else None


def rollback(to_run_id: str):
    promote(to_run_id)


def load_production_model_path() -> Optional[Path]:
    run_id = current_production_run()
    if not run_id:
        return None
    path = runs_dir() / run_id / "model.pkl"
    return path if path.exists() else None
