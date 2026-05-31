"""Batch-triggered continuous forecasting training cycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.forecasting.feeder_fingerprint import build_fingerprint_database
from app.forecasting.schema import ensure_canonical_history
from app.forecasting.tree_forecaster import TreeForecaster
from app.simulator.ami_history_generator import generate_ami_history, load_ami_history, save_ami_history
from app.simulator.continuous_ami_generator import append_continuous_ami_history


def generate_continuous_history(
    *,
    history_path: Path | None = None,
    fingerprint_path: Path | None = None,
    steps: int = 4,
    seed: int | None = None,
) -> dict[str, Any]:
    """Append synthetic continuous rows and rebuild fingerprints."""

    settings = get_settings()
    target_history_path = history_path or settings.ami_history_path
    target_fingerprint_path = fingerprint_path or settings.forecast_fingerprint_path
    history = load_ami_history(target_history_path)
    if history is None:
        history = generate_ami_history(days=3, seed=seed)

    before_rows = len(history)
    updated = append_continuous_ami_history(history, steps=steps, seed=seed)
    save_ami_history(updated, target_history_path)
    fingerprint = build_fingerprint_database(updated)
    target_fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint.to_csv(target_fingerprint_path, index=False)

    return {
        "status": "ok",
        "history_path": str(target_history_path),
        "fingerprint_path": str(target_fingerprint_path),
        "generated_rows": int(len(updated) - before_rows),
        "history_rows": int(len(updated)),
        "fingerprint_rows": int(len(fingerprint)),
        "latest_timestamp": pd.to_datetime(updated["timestamp"], utc=True).max().isoformat(),
    }


def run_continuous_training_cycle(
    *,
    history_path: Path | None = None,
    fingerprint_path: Path | None = None,
    artifact_dir: Path | None = None,
    steps: int = 4,
    seed: int | None = None,
    model: str = "tree",
    horizon: str = "1h",
    entity_type: str = "feeder",
    entity_id: str = "FD_RES_01",
) -> dict[str, Any]:
    """Generate the next AMI slice, rebuild fingerprints, and train a model."""

    if model.strip().lower() != "tree":
        raise ValueError("Only the tree model is supported for continuous training in this phase.")

    settings = get_settings()
    target_history_path = history_path or settings.ami_history_path
    target_artifact_dir = artifact_dir or settings.forecast_artifact_dir
    generation = generate_continuous_history(
        history_path=target_history_path,
        fingerprint_path=fingerprint_path,
        steps=steps,
        seed=seed,
    )
    history = ensure_canonical_history(pd.read_csv(target_history_path))
    training = TreeForecaster(model_dir=target_artifact_dir, n_estimators=10, n_jobs=1).train(
        history,
        entity_type=entity_type,
        entity_id=entity_id,
        horizon=horizon,
    )
    return {
        **generation,
        "model": "tree",
        "horizon": horizon,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "training": training,
    }


def get_continuous_training_status(
    *,
    history_path: Path | None = None,
    fingerprint_path: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Return local continuous-training state without starting a cycle."""

    settings = get_settings()
    target_history_path = history_path or settings.ami_history_path
    target_fingerprint_path = fingerprint_path or settings.forecast_fingerprint_path
    target_artifact_dir = artifact_dir or settings.forecast_artifact_dir
    history_exists = target_history_path.exists()
    latest_timestamp = None
    history_rows = 0
    if history_exists:
        history = ensure_canonical_history(pd.read_csv(target_history_path))
        history_rows = int(len(history))
        latest_timestamp = pd.to_datetime(history["timestamp"], utc=True).max().isoformat()

    return {
        "status": "ok",
        "history_exists": history_exists,
        "fingerprint_exists": target_fingerprint_path.exists(),
        "artifact_dir_exists": target_artifact_dir.exists(),
        "history_path": str(target_history_path),
        "fingerprint_path": str(target_fingerprint_path),
        "artifact_dir": str(target_artifact_dir),
        "history_rows": history_rows,
        "latest_timestamp": latest_timestamp,
    }
