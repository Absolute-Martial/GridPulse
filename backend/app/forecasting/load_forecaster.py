"""Compatibility wrapper around the modular forecasting interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from app.forecasting.features import FORECAST_INPUT_COLUMNS, build_zone_feature_frame
from app.forecasting.model_registry import (
    AUTO_MODEL_NAME,
    ForecastModelNotFoundError,
    ForecastTrainingError,
    ForecastZoneError,
    get_forecaster,
    resolve_model_directory,
    resolve_model_name,
)


def prepare_training_frame(sensor_frame: pd.DataFrame) -> pd.DataFrame:
    return build_zone_feature_frame(sensor_frame, dropna_lags=False)


def normalize_zone_id(zone_id: str) -> str:
    from app.forecasting.features import normalize_zone_id as _normalize_zone_id

    return _normalize_zone_id(zone_id)


@dataclass
class LoadForecaster:
    """Backwards-compatible forecaster wrapper for existing callers."""

    model_path: Path | None = None
    model_name: str = AUTO_MODEL_NAME

    def __post_init__(self) -> None:
        self.legacy_bundle_path = self.model_path if self.model_path and self.model_path.suffix else None
        self.model_dir = resolve_model_directory(self.model_path)

    def train(self, sensor_frame: pd.DataFrame) -> dict[str, Any]:
        zone_ids = sorted(set(sensor_frame["zone_id"].astype(str)))
        horizons = [1, 6, 24]
        results = []
        for zone_id in zone_ids:
            for horizon in horizons:
                resolved_model = resolve_model_name(self.model_name, horizon)
                forecaster = get_forecaster(self.model_name, model_dir=self.model_dir, horizon=horizon)
                result = forecaster.train(sensor_frame, zone_id=zone_id, horizon=horizon)
                result["resolved_model"] = resolved_model
                results.append(result)

        compatibility_bundle = {
            "model_name": self.model_name,
            "model_dir": str(self.model_dir),
            "artifacts": [result["artifact_path"] for result in results],
            "resolved_models": sorted({result["resolved_model"] for result in results}),
            "metrics": results[0]["metrics"] if results else {},
        }
        if self.legacy_bundle_path is not None:
            self.legacy_bundle_path.parent.mkdir(parents=True, exist_ok=True)
            with self.legacy_bundle_path.open("wb") as artifact_file:
                pickle.dump(compatibility_bundle, artifact_file)

        return {
            "artifact_path": str(self.legacy_bundle_path or results[0]["artifact_path"]),
            "train_rows": sum(int(result["train_rows"]) for result in results),
            "test_rows": sum(int(result["test_rows"]) for result in results),
            "metrics": results[0]["metrics"] if results else {},
        }

    def run(
        self,
        zone_id: str,
        horizon: int,
        history_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        forecaster = get_forecaster(self.model_name, model_dir=self.model_dir, horizon=horizon)
        try:
            return forecaster.predict(history_frame, zone_id=zone_id, horizon=horizon)
        except ForecastModelNotFoundError:
            forecaster.train(history_frame, zone_id=zone_id, horizon=horizon)
            return forecaster.predict(history_frame, zone_id=zone_id, horizon=horizon)

    def evaluate(
        self,
        sensor_frame: pd.DataFrame,
        zone_id: str,
        horizon: int,
    ) -> dict[str, Any]:
        forecaster = get_forecaster(self.model_name, model_dir=self.model_dir, horizon=horizon)
        try:
            return forecaster.evaluate(sensor_frame, zone_id=zone_id, horizon=horizon)
        except ForecastModelNotFoundError:
            forecaster.train(sensor_frame, zone_id=zone_id, horizon=horizon)
            return forecaster.evaluate(sensor_frame, zone_id=zone_id, horizon=horizon)

    def load_artifact(self) -> dict[str, Any]:
        if self.legacy_bundle_path is not None and self.legacy_bundle_path.exists():
            with self.legacy_bundle_path.open("rb") as artifact_file:
                return pickle.load(artifact_file)
        pattern = "*_*_*.*" if self.model_name == AUTO_MODEL_NAME else f"{self.model_name}_*.*"
        artifact_candidates = sorted(self.model_dir.glob(pattern))
        if not artifact_candidates:
            raise ForecastModelNotFoundError("Forecast model artifact not found.")
        candidate = artifact_candidates[0]
        resolved_model_name = candidate.stem.rsplit("_", 2)[0]
        horizon = int(candidate.stem.rsplit("_", 1)[-1])
        return get_forecaster(resolved_model_name, model_dir=self.model_dir, horizon=horizon).load(candidate).artifact  # type: ignore[return-value]


__all__ = [
    "FORECAST_INPUT_COLUMNS",
    "ForecastModelNotFoundError",
    "ForecastTrainingError",
    "ForecastZoneError",
    "LoadForecaster",
    "normalize_zone_id",
    "prepare_training_frame",
]
