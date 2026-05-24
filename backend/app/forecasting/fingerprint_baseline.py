"""Fingerprint-baseline forecasting model for canonical 15-minute history."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from app.forecasting.base import BaseForecaster
from app.forecasting.feeder_fingerprint import build_fingerprint_database
from app.forecasting.features import filter_target_history
from app.forecasting.metrics import calculate_regression_metrics
from app.forecasting.schema import (
    CANONICAL_RESOLUTION_MINUTES,
    ForecastFingerprintError,
    ForecastUntrainedModelError,
    normalize_horizon,
)


@dataclass
class FingerprintBaselineForecaster(BaseForecaster):
    model_dir: Path | None = None
    model_name: str = "fingerprint_baseline"
    artifact: dict[str, Any] | None = field(default=None, init=False)

    def fit_fingerprint(self, fingerprint_database: pd.DataFrame) -> None:
        self.artifact = {
            "model_name": self.model_name,
            "fingerprint_database": fingerprint_database.copy(),
        }

    def train(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        normalized_horizon, _ = normalize_horizon(horizon)
        history = filter_target_history(dataframe, entity_type=entity_type, entity_id=entity_id)
        fingerprint_database = build_fingerprint_database(history)
        self.artifact = {
            "model_name": self.model_name,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "horizon": normalized_horizon,
            "fingerprint_database": fingerprint_database,
        }
        artifact_path = self._artifact_path(entity_type, entity_id, normalized_horizon)
        self.save(artifact_path)
        return {
            "artifact_path": str(artifact_path),
            "train_rows": int(len(history)),
            "test_rows": 0,
            "metrics": {},
        }

    def predict(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        normalized_horizon, horizon_steps = normalize_horizon(horizon)
        history = filter_target_history(dataframe, entity_type=entity_type, entity_id=entity_id)
        if self.artifact is None:
            self.load(self._artifact_path(entity_type, entity_id, normalized_horizon))

        fingerprint_database = self._fingerprint_database()
        anchor = pd.to_datetime(history["timestamp"], utc=True).iloc[-1]
        slot_predictions: list[dict[str, Any]] = []

        for step in range(1, horizon_steps + 1):
            prediction_timestamp = anchor + pd.Timedelta(minutes=CANONICAL_RESOLUTION_MINUTES * step)
            slot_index = int(prediction_timestamp.hour * 4 + (prediction_timestamp.minute // 15))
            row = fingerprint_database[
                (fingerprint_database["entity_type"] == entity_type)
                & (fingerprint_database["entity_id"] == entity_id)
                & (fingerprint_database["day_of_week"] == prediction_timestamp.day_of_week)
                & (fingerprint_database["slot_index"] == slot_index)
            ]
            if row.empty:
                raise ForecastFingerprintError(
                    f"Missing fingerprint for {entity_type}:{entity_id} at day={prediction_timestamp.day_of_week}, slot={slot_index}."
                )
            matched = row.iloc[0]
            slot_predictions.append(
                {
                    "timestamp": prediction_timestamp.isoformat(),
                    "predicted_load_kw": round(float(matched["fingerprint_mean_kw"]), 3),
                    "p10_kw": round(float(matched["fingerprint_p10_kw"]), 3),
                    "p90_kw": round(float(matched["fingerprint_p90_kw"]), 3),
                    "fingerprint_mean_kw": round(float(matched["fingerprint_mean_kw"]), 3),
                }
            )

        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "model_name": self.model_name,
            "horizon": normalized_horizon,
            "horizon_steps": horizon_steps,
            "latest_input_timestamp": anchor.isoformat(),
            "slot_predictions": slot_predictions,
        }

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        normalized_horizon, horizon_steps = normalize_horizon(horizon)
        history = filter_target_history(dataframe, entity_type=entity_type, entity_id=entity_id)
        if len(history) <= horizon_steps:
            raise ValueError("Not enough rows to evaluate fingerprint baseline.")

        train_history = history.iloc[:-horizon_steps].copy()
        actual = history["load_kw"].iloc[-horizon_steps:].astype(float).tolist()
        self.artifact = {
            "model_name": self.model_name,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "horizon": normalized_horizon,
            "fingerprint_database": build_fingerprint_database(train_history),
        }
        predicted = [
            item["predicted_load_kw"]
            for item in self.predict(train_history, entity_type=entity_type, entity_id=entity_id, horizon=normalized_horizon)[
                "slot_predictions"
            ]
        ]
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "horizon": normalized_horizon,
            "metrics": calculate_regression_metrics(actual, predicted),
        }

    def save(self, path: Path) -> None:
        if self.artifact is None:
            raise ForecastUntrainedModelError("Fingerprint baseline artifact is not available.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as artifact_file:
            pickle.dump(self.artifact, artifact_file)

    def load(self, path: Path) -> "FingerprintBaselineForecaster":
        if not path.exists():
            raise ForecastUntrainedModelError(f"Fingerprint baseline artifact not found at {path}.")
        with path.open("rb") as artifact_file:
            self.artifact = pickle.load(artifact_file)
        return self

    def _artifact_path(self, entity_type: str, entity_id: str, horizon: str) -> Path:
        directory = self.model_dir or (Path(__file__).resolve().parents[3] / "data" / "models" / "forecasting")
        directory.mkdir(parents=True, exist_ok=True)
        safe_entity_id = entity_id.replace("/", "_")
        return directory / f"{self.model_name}_{entity_type}_{safe_entity_id}_{horizon}.pkl"

    def _fingerprint_database(self) -> pd.DataFrame:
        if self.artifact is None or "fingerprint_database" not in self.artifact:
            raise ForecastUntrainedModelError("Fingerprint baseline artifact is not loaded.")
        return pd.DataFrame(self.artifact["fingerprint_database"])

