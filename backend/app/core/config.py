"""Runtime configuration for the GridPulse backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    synthetic_csv_path: Path
    forecast_model_path: Path
    ami_history_path: Path
    forecast_fingerprint_path: Path
    forecast_artifact_dir: Path
    anomaly_model_path: Path
    gnn_model_path: Path


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[3]

    env_path = os.getenv("GRIDPULSE_SYNTHETIC_CSV_PATH")
    if env_path:
        synthetic_csv_path = Path(env_path).expanduser().resolve()
    else:
        synthetic_csv_path = repo_root / "data" / "synthetic" / "sensor_data.csv"

    forecast_env_path = os.getenv("GRIDPULSE_FORECAST_MODEL_PATH")
    if forecast_env_path:
        forecast_model_path = Path(forecast_env_path).expanduser().resolve()
    else:
        forecast_model_path = repo_root / "data" / "models" / "load_forecaster.pkl"

    ami_history_env_path = os.getenv("GRIDPULSE_AMI_HISTORY_PATH")
    if ami_history_env_path:
        ami_history_path = Path(ami_history_env_path).expanduser().resolve()
    else:
        ami_history_path = repo_root / "data" / "forecasting" / "ami_history.csv"

    fingerprint_env_path = os.getenv("GRIDPULSE_FORECAST_FINGERPRINT_PATH")
    if fingerprint_env_path:
        forecast_fingerprint_path = Path(fingerprint_env_path).expanduser().resolve()
    else:
        forecast_fingerprint_path = repo_root / "data" / "forecasting" / "fingerprints.csv"

    artifact_env_path = os.getenv("GRIDPULSE_FORECAST_ARTIFACT_DIR")
    if artifact_env_path:
        forecast_artifact_dir = Path(artifact_env_path).expanduser().resolve()
    else:
        forecast_artifact_dir = repo_root / "data" / "models" / "forecasting"

    anomaly_env_path = os.getenv("GRIDPULSE_ANOMALY_MODEL_PATH")
    if anomaly_env_path:
        anomaly_model_path = Path(anomaly_env_path).expanduser().resolve()
    else:
        anomaly_model_path = repo_root / "data" / "models" / "anomaly_detector.pkl"

    gnn_env_path = os.getenv("GRIDPULSE_GNN_MODEL_PATH")
    if gnn_env_path:
        gnn_model_path = Path(gnn_env_path).expanduser().resolve()
    else:
        gnn_model_path = repo_root / "data" / "models" / "gnn" / "grid_risk_gnn.pt"

    return Settings(
        synthetic_csv_path=synthetic_csv_path,
        forecast_model_path=forecast_model_path,
        ami_history_path=ami_history_path,
        forecast_fingerprint_path=forecast_fingerprint_path,
        forecast_artifact_dir=forecast_artifact_dir,
        anomaly_model_path=anomaly_model_path,
        gnn_model_path=gnn_model_path,
    )
