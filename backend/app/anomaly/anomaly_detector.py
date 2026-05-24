"""IsolationForest-based anomaly detection for GridPulse telemetry."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.config import get_settings

FEATURE_COLUMNS: tuple[str, ...] = (
    "voltage_pu",
    "frequency_hz",
    "load_mw",
    "line_flow_mw",
    "line_capacity_mw",
    "loading_percent",
    "temperature_c",
    "storage_soc_percent",
)

REQUIRED_OUTPUT_KEYS: tuple[str, ...] = (
    "timestamp",
    "sensor_id",
    "anomaly_score",
    "anomaly_type",
    "severity",
    "affected_zone",
    "explanation_text",
)


class AnomalyTrainingError(ValueError):
    """Raised when the detector cannot train from the available data."""


class AnomalyModelNotFoundError(FileNotFoundError):
    """Raised when the anomaly model artifact is unavailable."""


@dataclass
class GridAnomalyDetector:
    """Train and apply anomaly detection to synthetic smart-grid telemetry."""

    model_path: Path | None = None

    def __post_init__(self) -> None:
        self.model_path = self.model_path or get_settings().anomaly_model_path

    def train(self, sensor_frame: pd.DataFrame) -> dict[str, Any]:
        prepared = prepare_anomaly_frame(sensor_frame)
        if len(prepared) < 48:
            raise AnomalyTrainingError("Not enough telemetry rows to train the anomaly detector.")

        model = IsolationForest(
            contamination=0.06,
            n_estimators=200,
            random_state=42,
        )
        model.fit(prepared.loc[:, FEATURE_COLUMNS])

        raw_scores = -model.decision_function(prepared.loc[:, FEATURE_COLUMNS])
        score_threshold = float(np.percentile(raw_scores, 95))
        zone_load_thresholds = (
            prepared.groupby("zone_id")["load_mw"]
            .agg(load_median="median", load_p95=lambda values: np.percentile(values, 95))
            .reset_index()
        )

        artifact = {
            "model": model,
            "feature_columns": list(FEATURE_COLUMNS),
            "score_threshold": max(score_threshold, 1e-6),
            "zone_load_thresholds": zone_load_thresholds.to_dict(orient="records"),
            "trained_at_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        self._save_artifact(artifact)

        return {
            "artifact_path": str(self.model_path),
            "train_rows": int(len(prepared)),
            "feature_count": len(FEATURE_COLUMNS),
        }

    def detect_latest(self, sensor_frame: pd.DataFrame) -> list[dict[str, Any]]:
        prepared = prepare_anomaly_frame(sensor_frame)
        latest_timestamp = prepared["timestamp"].max()
        latest_frame = prepared[prepared["timestamp"] == latest_timestamp].copy()
        return self._detect(latest_frame)

    def detect_history(self, sensor_frame: pd.DataFrame) -> list[dict[str, Any]]:
        prepared = prepare_anomaly_frame(sensor_frame)
        return self._detect(prepared)

    def load_artifact(self) -> dict[str, Any]:
        if self.model_path is None or not self.model_path.exists():
            raise AnomalyModelNotFoundError("Anomaly detector artifact not found.")

        with self.model_path.open("rb") as artifact_file:
            return pickle.load(artifact_file)

    def _detect(self, prepared: pd.DataFrame) -> list[dict[str, Any]]:
        artifact = self.load_artifact()
        model: IsolationForest = artifact["model"]
        threshold = float(artifact["score_threshold"])
        load_thresholds = {
            item["zone_id"]: item for item in artifact["zone_load_thresholds"]
        }

        working = prepared.copy()
        working["raw_anomaly_score"] = -model.decision_function(working.loc[:, FEATURE_COLUMNS])
        working["model_flag"] = model.predict(working.loc[:, FEATURE_COLUMNS]) == -1

        anomalies: list[dict[str, Any]] = []
        for _, row in working.iterrows():
            anomaly_type = classify_anomaly(row, load_thresholds.get(str(row["zone_id"])))
            score = normalized_anomaly_score(float(row["raw_anomaly_score"]), threshold)
            flagged = bool(row["model_flag"]) or anomaly_type is not None
            if not flagged:
                continue

            final_type = anomaly_type or "possible sensor fault"
            severity = classify_severity(final_type, row, score)
            explanation = build_explanation(final_type, row, score)
            anomalies.append(
                {
                    "timestamp": pd.to_datetime(row["timestamp"], utc=True).isoformat(),
                    "sensor_id": str(row["sensor_id"]),
                    "anomaly_score": round(score, 4),
                    "anomaly_type": final_type,
                    "severity": severity,
                    "affected_zone": str(row["zone_id"]),
                    "explanation_text": explanation,
                }
            )

        anomalies.sort(key=lambda item: (item["timestamp"], item["anomaly_score"]), reverse=True)
        return anomalies

    def _save_artifact(self, artifact: dict[str, Any]) -> None:
        if self.model_path is None:
            raise AnomalyModelNotFoundError("Anomaly model path is not configured.")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with self.model_path.open("wb") as artifact_file:
            pickle.dump(artifact, artifact_file)


def prepare_anomaly_frame(sensor_frame: pd.DataFrame) -> pd.DataFrame:
    frame = sensor_frame.copy()
    if frame.empty:
        raise AnomalyTrainingError("Sensor history is empty.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["loading_percent"] = np.where(
        frame["line_capacity_mw"].astype(float) > 0,
        (frame["line_flow_mw"].astype(float) / frame["line_capacity_mw"].astype(float)) * 100.0,
        0.0,
    )
    return frame


def classify_anomaly(row: pd.Series, zone_stats: dict[str, Any] | None) -> str | None:
    voltage = float(row["voltage_pu"])
    frequency = float(row["frequency_hz"])
    load_mw = float(row["load_mw"])
    loading_percent = float(row["loading_percent"])
    line_capacity = float(row["line_capacity_mw"])
    fault_status = str(row.get("fault_status", "normal"))

    if fault_status == "fault" or line_capacity <= 0.0:
        return "possible sensor fault"
    if loading_percent > 95.0:
        return "line overload"
    if frequency < 49.5 or frequency > 50.5:
        return "frequency anomaly"
    if voltage < 0.95 or voltage > 1.05:
        return "voltage anomaly"
    if zone_stats is not None:
        threshold = max(float(zone_stats["load_p95"]), float(zone_stats["load_median"]) * 1.35)
        if load_mw > threshold:
            return "load spike"
    return None


def normalized_anomaly_score(raw_score: float, threshold: float) -> float:
    baseline = max(threshold, 1e-6)
    return max(raw_score / baseline, 0.0)


def classify_severity(anomaly_type: str, row: pd.Series, anomaly_score: float) -> str:
    if anomaly_type == "possible sensor fault":
        return "high"
    if anomaly_type == "line overload":
        return "high" if float(row["loading_percent"]) >= 100.0 else "medium"
    if anomaly_type == "frequency anomaly":
        return "high" if abs(float(row["frequency_hz"]) - 50.0) >= 1.0 else "medium"
    if anomaly_type == "voltage anomaly":
        return "high" if abs(float(row["voltage_pu"]) - 1.0) >= 0.08 else "medium"
    if anomaly_score >= 1.4:
        return "medium"
    return "low"


def build_explanation(anomaly_type: str, row: pd.Series, anomaly_score: float) -> str:
    if anomaly_type == "voltage anomaly":
        return (
            f"Voltage moved to {float(row['voltage_pu']):.3f} pu in {row['zone_id']}, "
            "outside the expected operating band."
        )
    if anomaly_type == "frequency anomaly":
        return (
            f"Frequency shifted to {float(row['frequency_hz']):.3f} Hz in {row['zone_id']}, "
            "indicating a grid balance disturbance."
        )
    if anomaly_type == "load spike":
        return (
            f"Load reached {float(row['load_mw']):.2f} MW in {row['zone_id']}, "
            "well above the trained zone baseline."
        )
    if anomaly_type == "line overload":
        return (
            f"Line {row['line_id']} is carrying {float(row['loading_percent']):.1f}% of capacity, "
            "indicating an overload condition."
        )
    return (
        f"Sensor {row['sensor_id']} shows an unusual multivariate pattern "
        f"(score {anomaly_score:.2f}) consistent with a possible sensor fault."
    )
