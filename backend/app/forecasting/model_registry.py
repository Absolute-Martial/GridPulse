"""Forecast model registry and baseline implementations."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.core.config import get_settings
from app.forecasting.base import BaseForecaster
from app.forecasting.datasets import build_training_dataset, extract_model_matrices, split_train_test
from app.forecasting.features import (
    MODEL_FEATURE_COLUMNS,
    append_prediction_row,
    build_recursive_feature_row,
    build_zone_feature_frame,
    build_zone_profiles,
    lookup_exogenous_profile,
    normalize_zone_id,
)
from app.forecasting.metrics import calculate_regression_metrics

AUTO_MODEL_NAME = "auto"
SUPPORTED_MODELS: tuple[str, ...] = ("moving_average", "random_forest", "tcn", "n_hits", AUTO_MODEL_NAME)
TARGET_SUPPORTED_MODELS: tuple[str, ...] = ("fingerprint_baseline", "tree", "tcn", "nhits")


class ForecastTrainingError(ValueError):
    """Raised when the forecaster cannot train from the available data."""


class ForecastModelNotFoundError(FileNotFoundError):
    """Raised when a requested forecast artifact is missing."""


class ForecastZoneError(ValueError):
    """Raised when a requested zone does not exist in the dataset."""


def list_models() -> list[str]:
    return list(SUPPORTED_MODELS)


def list_target_models() -> list[str]:
    return list(TARGET_SUPPORTED_MODELS)


def resolve_model_name(model_name: str, horizon: int | None = None) -> str:
    normalized = model_name.strip().lower()
    if normalized != AUTO_MODEL_NAME:
        return normalized
    if horizon is None:
        raise KeyError("A forecast horizon is required when using the auto forecasting model selector.")
    if int(horizon) in {1, 6}:
        return "tcn"
    if int(horizon) == 24:
        return "n_hits"
    raise KeyError(f"No default forecasting model is configured for horizon {horizon}.")


def get_forecaster(model_name: str, model_dir: Path | None = None, horizon: int | None = None) -> BaseForecaster:
    normalized = resolve_model_name(model_name, horizon)
    if normalized == "moving_average":
        return MovingAverageForecaster(model_dir=model_dir)
    if normalized == "random_forest":
        return RandomForestForecaster(model_dir=model_dir)
    if normalized == "tcn":
        from app.forecasting.tcn import TCNForecaster

        return TCNForecaster(model_dir=model_dir)
    if normalized == "n_hits":
        from app.forecasting.n_hits import NHiTSForecaster

        return NHiTSForecaster(model_dir=model_dir)
    raise KeyError(f"Unsupported forecasting model: {model_name}")


def get_target_forecaster(model_name: str, model_dir: Path | None = None) -> BaseForecaster:
    normalized = model_name.strip().lower()
    if normalized == "fingerprint_baseline":
        from app.forecasting.fingerprint_baseline import FingerprintBaselineForecaster

        return FingerprintBaselineForecaster(model_dir=model_dir)
    if normalized == "tree":
        from app.forecasting.tree_forecaster import TreeForecaster

        return TreeForecaster(model_dir=model_dir)
    if normalized == "tcn":
        from app.forecasting.tcn import TCNForecaster

        return TCNForecaster(model_dir=model_dir)
    if normalized == "nhits":
        from app.forecasting.nhits import NHiTSForecaster

        return NHiTSForecaster(model_dir=model_dir)
    raise KeyError(f"Unsupported target forecasting model: {model_name}")


def resolve_model_directory(model_dir: Path | None = None) -> Path:
    if model_dir is not None:
        if model_dir.suffix:
            return model_dir.parent / "forecasting"
        return model_dir

    base_path = get_settings().forecast_model_path
    if base_path.suffix:
        return base_path.parent / "forecasting"
    return base_path / "forecasting"


def model_artifact_path(model_name: str, zone_id: str, horizon: int, model_dir: Path | None = None) -> Path:
    directory = resolve_model_directory(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    normalized_zone_id = normalize_zone_id(zone_id)
    resolved_name = resolve_model_name(model_name, horizon)
    extension = ".pt" if resolved_name in {"tcn", "n_hits"} else ".pkl"
    return directory / f"{resolved_name}_{normalized_zone_id}_{int(horizon)}{extension}"


@dataclass
class _BaseOfflineForecaster(BaseForecaster):
    model_name: str
    model_dir: Path | None = None
    artifact: dict[str, Any] | None = field(default=None, init=False)

    def save(self, path: Path) -> None:
        if self.artifact is None:
            raise ForecastModelNotFoundError("No trained forecast artifact is available to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as artifact_file:
            pickle.dump(self.artifact, artifact_file)

    def load(self, path: Path) -> "_BaseOfflineForecaster":
        if not path.exists():
            raise ForecastModelNotFoundError(f"Forecast artifact not found at {path}")
        with path.open("rb") as artifact_file:
            self.artifact = pickle.load(artifact_file)
        return self

    def evaluate(self, dataframe: pd.DataFrame, zone_id: str, horizon: int) -> dict[str, Any]:
        normalized_zone_id = normalize_zone_id(zone_id)
        artifact_path = model_artifact_path(self.model_name, normalized_zone_id, horizon, self.model_dir)
        self.load(artifact_path)
        zone_frame = build_zone_feature_frame(dataframe, zone_id=normalized_zone_id, dropna_lags=False)
        if len(zone_frame) <= horizon + 24:
            raise ForecastTrainingError("Not enough hourly rows to evaluate the requested horizon.")

        history_frame = zone_frame.iloc[:-horizon].copy()
        actual_values = zone_frame["load_mw"].iloc[-horizon:].astype(float).tolist()
        prediction = self._predict_from_zone_history(history_frame, normalized_zone_id, horizon)
        predicted_values = [item["predicted_load_mw"] for item in prediction["forecast_series"]]

        return {
            "zone_id": normalized_zone_id,
            "model": self.model_name,
            "horizon": horizon,
            "metrics": calculate_regression_metrics(actual_values, predicted_values),
            "actual": actual_values,
            "predicted": predicted_values,
        }

    def train(self, dataframe: pd.DataFrame, zone_id: str, horizon: int) -> dict[str, Any]:
        raise NotImplementedError

    def predict(self, dataframe: pd.DataFrame, zone_id: str, horizon: int) -> dict[str, Any]:
        normalized_zone_id = normalize_zone_id(zone_id)
        artifact_path = model_artifact_path(self.model_name, normalized_zone_id, horizon, self.model_dir)
        self.load(artifact_path)
        zone_history = build_zone_feature_frame(dataframe, zone_id=normalized_zone_id, dropna_lags=False)
        if zone_history.empty:
            raise ForecastZoneError(f"Zone {normalized_zone_id} is not present in the sensor history.")
        return self._predict_from_zone_history(zone_history, normalized_zone_id, horizon)

    def _predict_from_zone_history(
        self,
        zone_history: pd.DataFrame,
        zone_id: str,
        horizon: int,
    ) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class MovingAverageForecaster(_BaseOfflineForecaster):
    model_name: str = "moving_average"

    def train(self, dataframe: pd.DataFrame, zone_id: str, horizon: int) -> dict[str, Any]:
        normalized_zone_id = normalize_zone_id(zone_id)
        dataset = build_training_dataset(dataframe, normalized_zone_id)
        if len(dataset) < 48:
            raise ForecastTrainingError("Not enough hourly training samples to train the forecaster.")

        train_frame, test_frame = split_train_test(dataset)
        if train_frame.empty or test_frame.empty:
            raise ForecastTrainingError("Training split produced an empty train or test partition.")

        window = 6 if horizon <= 6 else 24
        predictions = (
            test_frame["lag_6"].fillna(test_frame["lag_3"]) if window == 6 else test_frame["rolling_mean_24"]
        ).fillna(test_frame["lag_1"])
        metrics = calculate_regression_metrics(test_frame["target_load_mw"], predictions)
        profiles = build_zone_profiles(build_zone_feature_frame(dataframe, zone_id=normalized_zone_id, dropna_lags=False))

        self.artifact = {
            "model_name": self.model_name,
            "zone_id": normalized_zone_id,
            "horizon": int(horizon),
            "window": window,
            "metrics": metrics,
            "profile_records": profiles.to_dict(orient="records"),
        }
        artifact_path = model_artifact_path(self.model_name, normalized_zone_id, horizon, self.model_dir)
        self.save(artifact_path)
        return {
            "artifact_path": str(artifact_path),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "metrics": metrics,
        }

    def _predict_from_zone_history(
        self,
        zone_history: pd.DataFrame,
        zone_id: str,
        horizon: int,
    ) -> dict[str, Any]:
        if self.artifact is None:
            raise ForecastModelNotFoundError("Forecast artifact is not loaded.")

        profiles = pd.DataFrame(self.artifact["profile_records"])
        working_history = zone_history.copy().reset_index(drop=True)
        forecast_rows: list[dict[str, Any]] = []
        window = int(self.artifact["window"])

        for _ in range(horizon):
            next_timestamp = pd.to_datetime(working_history["timestamp"].iloc[-1], utc=True) + pd.Timedelta(hours=1)
            exogenous = lookup_exogenous_profile(profiles, next_timestamp, working_history)
            predicted_load = float(working_history["load_mw"].iloc[-window:].mean())
            lower_bound = max(predicted_load * 0.92, 0.0)
            upper_bound = predicted_load * 1.08
            forecast_rows.append(
                {
                    "timestamp": next_timestamp.isoformat(),
                    "predicted_load_mw": round(predicted_load, 3),
                    "lower_mw": round(lower_bound, 3),
                    "upper_mw": round(upper_bound, 3),
                }
            )
            working_history = append_prediction_row(
                working_history,
                next_timestamp,
                zone_id,
                predicted_load,
                exogenous,
            )

        return _format_prediction_payload(
            zone_id=zone_id,
            horizon=horizon,
            forecast_rows=forecast_rows,
            metrics=self.artifact["metrics"],
            model_name=self.model_name,
        )


@dataclass
class RandomForestForecaster(_BaseOfflineForecaster):
    model_name: str = "random_forest"

    def train(self, dataframe: pd.DataFrame, zone_id: str, horizon: int) -> dict[str, Any]:
        normalized_zone_id = normalize_zone_id(zone_id)
        dataset = build_training_dataset(dataframe, normalized_zone_id)
        if len(dataset) < 48:
            raise ForecastTrainingError("Not enough hourly training samples to train the forecaster.")

        train_frame, test_frame = split_train_test(dataset)
        if train_frame.empty or test_frame.empty:
            raise ForecastTrainingError("Training split produced an empty train or test partition.")

        x_train, y_train = extract_model_matrices(train_frame)
        x_test, y_test = extract_model_matrices(test_frame)

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        metrics = calculate_regression_metrics(y_test, predictions)
        residual_std = float(np.std(y_test.to_numpy() - predictions, ddof=0))
        profiles = build_zone_profiles(build_zone_feature_frame(dataframe, zone_id=normalized_zone_id, dropna_lags=False))

        self.artifact = {
            "model_name": self.model_name,
            "zone_id": normalized_zone_id,
            "horizon": int(horizon),
            "model": model,
            "metrics": metrics,
            "residual_std": residual_std,
            "profile_records": profiles.to_dict(orient="records"),
        }
        artifact_path = model_artifact_path(self.model_name, normalized_zone_id, horizon, self.model_dir)
        self.save(artifact_path)
        return {
            "artifact_path": str(artifact_path),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "metrics": metrics,
        }

    def _predict_from_zone_history(
        self,
        zone_history: pd.DataFrame,
        zone_id: str,
        horizon: int,
    ) -> dict[str, Any]:
        if self.artifact is None:
            raise ForecastModelNotFoundError("Forecast artifact is not loaded.")

        profiles = pd.DataFrame(self.artifact["profile_records"])
        model: RandomForestRegressor = self.artifact["model"]
        residual_std = max(float(self.artifact.get("residual_std", 0.05)), 0.05)
        working_history = zone_history.copy().reset_index(drop=True)
        forecast_rows: list[dict[str, Any]] = []

        for _ in range(horizon):
            next_timestamp = pd.to_datetime(working_history["timestamp"].iloc[-1], utc=True) + pd.Timedelta(hours=1)
            exogenous = lookup_exogenous_profile(profiles, next_timestamp, working_history)
            feature_row = build_recursive_feature_row(working_history, next_timestamp, exogenous)
            predicted_load = max(0.0, float(model.predict(feature_row.loc[:, MODEL_FEATURE_COLUMNS])[0]))
            lower_bound = max(predicted_load - (1.96 * residual_std), 0.0)
            upper_bound = predicted_load + (1.96 * residual_std)
            forecast_rows.append(
                {
                    "timestamp": next_timestamp.isoformat(),
                    "predicted_load_mw": round(predicted_load, 3),
                    "lower_mw": round(lower_bound, 3),
                    "upper_mw": round(upper_bound, 3),
                }
            )
            working_history = append_prediction_row(
                working_history,
                next_timestamp,
                zone_id,
                predicted_load,
                exogenous,
            )

        return _format_prediction_payload(
            zone_id=zone_id,
            horizon=horizon,
            forecast_rows=forecast_rows,
            metrics=self.artifact["metrics"],
            model_name=self.model_name,
        )


def _format_prediction_payload(
    zone_id: str,
    horizon: int,
    forecast_rows: list[dict[str, Any]],
    metrics: dict[str, float],
    model_name: str,
) -> dict[str, Any]:
    return {
        "zone_id": zone_id,
        "requested_horizon_hours": horizon,
        "forecasts": {
            "1h": forecast_rows[min(0, len(forecast_rows) - 1)]["predicted_load_mw"],
            "6h": forecast_rows[min(5, len(forecast_rows) - 1)]["predicted_load_mw"],
            "24h": forecast_rows[min(23, len(forecast_rows) - 1)]["predicted_load_mw"],
        },
        "peak_load_estimate_mw": round(max(item["predicted_load_mw"] for item in forecast_rows), 3),
        "confidence_band": {
            "lower_mw": round(min(item["lower_mw"] for item in forecast_rows), 3),
            "upper_mw": round(max(item["upper_mw"] for item in forecast_rows), 3),
        },
        "forecast_series": forecast_rows,
        "metrics": metrics,
        "model_type": model_name,
    }
