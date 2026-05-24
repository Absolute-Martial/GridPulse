"""Tree-based forecasting model for canonical 15-minute target history."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.core.config import get_settings
from app.forecasting.base import BaseForecaster
from app.forecasting.features import (
    TARGET_MODEL_FEATURE_COLUMNS,
    build_target_feature_frame,
    filter_target_history,
)
from app.forecasting.metrics import calculate_regression_metrics
from app.forecasting.schema import ForecastUntrainedModelError, normalize_horizon


@dataclass
class TreeForecaster(BaseForecaster):
    model_dir: Path | None = None
    model_name: str = "tree"
    random_state: int = 7
    n_estimators: int = 300
    min_samples_leaf: int = 2
    artifact: dict[str, Any] | None = field(default=None, init=False)

    def train(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        normalized_horizon, horizon_steps = normalize_horizon(horizon)
        dataset = self._build_training_dataset(dataframe, entity_type, entity_id, horizon_steps)
        if len(dataset) < 32:
            raise ValueError("Not enough target windows to train the tree forecaster.")

        split_index = max(1, int(len(dataset) * 0.8))
        if split_index >= len(dataset):
            split_index = len(dataset) - 1
        train_frame = dataset.iloc[:split_index].copy()
        test_frame = dataset.iloc[split_index:].copy()

        x_train = train_frame.loc[:, list(TARGET_MODEL_FEATURE_COLUMNS)]
        y_train = np.vstack(train_frame["target_sequence"].to_list())
        x_test = test_frame.loc[:, list(TARGET_MODEL_FEATURE_COLUMNS)]
        y_test = np.vstack(test_frame["target_sequence"].to_list())

        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        estimator.fit(x_train, y_train)
        test_predictions = estimator.predict(x_test)
        metrics = calculate_regression_metrics(y_test.flatten(), test_predictions.flatten())
        residuals = y_test - test_predictions
        lower_residual = float(np.quantile(residuals, 0.10))
        upper_residual = float(np.quantile(residuals, 0.90))

        self.artifact = {
            "model_name": self.model_name,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "horizon": normalized_horizon,
            "horizon_steps": horizon_steps,
            "feature_columns": list(TARGET_MODEL_FEATURE_COLUMNS),
            "model": estimator,
            "metrics": metrics,
            "lower_residual": lower_residual,
            "upper_residual": upper_residual,
        }
        artifact_path = self._artifact_path(entity_type, entity_id, normalized_horizon)
        self.save(artifact_path)
        return {
            "artifact_path": str(artifact_path),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "metrics": metrics,
        }

    def predict(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        normalized_horizon, horizon_steps = normalize_horizon(horizon)
        if self.artifact is None:
            self.load(self._artifact_path(entity_type, entity_id, normalized_horizon))

        feature_frame = build_target_feature_frame(
            dataframe,
            entity_type=entity_type,
            entity_id=entity_id,
            lookback_steps=480,
            horizon_steps=1,
        )
        latest_row = feature_frame.iloc[-1]
        model = self._artifact_model()
        raw_prediction = model.predict(pd.DataFrame([latest_row.loc[list(TARGET_MODEL_FEATURE_COLUMNS)]]))[0]
        anchor = pd.to_datetime(latest_row["timestamp"], utc=True)
        lower_residual = float(self.artifact["lower_residual"])
        upper_residual = float(self.artifact["upper_residual"])
        slot_predictions: list[dict[str, Any]] = []

        for index, predicted_value in enumerate(raw_prediction, start=1):
            timestamp = anchor + pd.Timedelta(minutes=15 * index)
            predicted_load = round(float(predicted_value), 3)
            p10 = round(float(max(predicted_value + lower_residual, 0.0)), 3)
            p90 = round(float(max(predicted_value + upper_residual, p10)), 3)
            slot_predictions.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "predicted_load_kw": predicted_load,
                    "p10_kw": p10,
                    "p90_kw": p90,
                    "fingerprint_mean_kw": round(float(latest_row["fingerprint_mean_kw"]), 3),
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
            "summary": {
                "confidence": self._confidence_label(len(filter_target_history(dataframe, entity_type, entity_id))),
            },
        }

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        normalized_horizon, horizon_steps = normalize_horizon(horizon)
        dataset = self._build_training_dataset(dataframe, entity_type, entity_id, horizon_steps)
        if len(dataset) < 2:
            raise ValueError("Not enough target windows to evaluate the tree forecaster.")
        if self.artifact is None:
            self.load(self._artifact_path(entity_type, entity_id, normalized_horizon))
        model = self._artifact_model()
        x_eval = dataset.loc[:, list(TARGET_MODEL_FEATURE_COLUMNS)]
        y_eval = np.vstack(dataset["target_sequence"].to_list())
        predictions = model.predict(x_eval)
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "horizon": normalized_horizon,
            "metrics": calculate_regression_metrics(y_eval.flatten(), predictions.flatten()),
        }

    def save(self, path: Path) -> None:
        if self.artifact is None:
            raise ForecastUntrainedModelError("Tree forecast artifact is not available.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as artifact_file:
            pickle.dump(self.artifact, artifact_file)

    def load(self, path: Path) -> "TreeForecaster":
        if not path.exists():
            raise ForecastUntrainedModelError(f"Tree forecast artifact not found at {path}.")
        with path.open("rb") as artifact_file:
            self.artifact = pickle.load(artifact_file)
        return self

    def _artifact_path(self, entity_type: str, entity_id: str, horizon: str) -> Path:
        directory = self.model_dir or get_settings().forecast_artifact_dir
        directory.mkdir(parents=True, exist_ok=True)
        safe_entity_id = entity_id.replace("/", "_")
        return directory / f"{self.model_name}_{entity_type}_{safe_entity_id}_{horizon}.pkl"

    def _build_training_dataset(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon_steps: int,
    ) -> pd.DataFrame:
        base = build_target_feature_frame(
            dataframe,
            entity_type=entity_type,
            entity_id=entity_id,
            lookback_steps=len(filter_target_history(dataframe, entity_type, entity_id)),
            horizon_steps=1,
        )
        rows: list[dict[str, Any]] = []
        base = base.reset_index(drop=True)
        for index in range(len(base) - horizon_steps):
            target_sequence = (
                base["load_kw"].iloc[index + 1 : index + 1 + horizon_steps].astype(float).tolist()
            )
            row = base.iloc[index].to_dict()
            row["target_sequence"] = target_sequence
            rows.append(row)
        return pd.DataFrame(rows)

    def _artifact_model(self) -> RandomForestRegressor:
        if self.artifact is None or "model" not in self.artifact:
            raise ForecastUntrainedModelError("Tree forecast artifact is not loaded.")
        return self.artifact["model"]

    def _confidence_label(self, history_rows: int) -> str:
        if history_rows >= 480:
            return "high"
        if history_rows >= 192:
            return "medium"
        return "low"
