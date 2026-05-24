"""Simplified N-HiTS-style forecaster for longer-horizon offline load prediction."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.forecasting.base import BaseForecaster
from app.forecasting.features import (
    MODEL_FEATURE_COLUMNS,
    build_zone_feature_frame,
    build_zone_profiles,
    lookup_exogenous_profile,
    normalize_zone_id,
)
from app.forecasting.metrics import calculate_regression_metrics

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
except ModuleNotFoundError:
    torch = None
    nn = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback dataset base when torch is unavailable."""

    class DataLoader:  # type: ignore[no-redef]
        """Fallback dataloader type when torch is unavailable."""

    class Subset:  # type: ignore[no-redef]
        """Fallback subset type when torch is unavailable."""


NHITS_FEATURE_COLUMNS: tuple[str, ...] = ("load_mw", *MODEL_FEATURE_COLUMNS)
DEFAULT_POOL_SIZES: tuple[int, ...] = (1, 2, 4)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for the n_hits forecasting model but is not installed.")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class WindowedTimeSeriesDataset(Dataset):
    """Create causal windows for one-step-ahead recursive forecasting."""

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        lookback: int = 24,
        target_offset: int = 1,
        feature_columns: tuple[str, ...] = NHITS_FEATURE_COLUMNS,
    ) -> None:
        self.lookback = int(lookback)
        self.target_offset = int(target_offset)
        self.feature_columns = tuple(feature_columns)

        working = feature_frame.sort_values("timestamp").reset_index(drop=True).copy()
        if working.empty:
            raise ValueError("Feature frame is empty.")
        if len(working) < self.lookback + self.target_offset:
            raise ValueError("Not enough rows to create N-HiTS sliding windows.")

        self.features = working.loc[:, self.feature_columns].astype(float).to_numpy(dtype=np.float32)
        self.targets = working["load_mw"].astype(float).to_numpy(dtype=np.float32)
        self.window_starts = list(range(0, len(working) - self.lookback - self.target_offset + 1))

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, index: int):
        _require_torch()
        start = self.window_starts[index]
        end = start + self.lookback
        target_index = end + self.target_offset - 1

        feature_window = self.features[start:end].T
        target_value = np.asarray([self.targets[target_index]], dtype=np.float32)
        return torch.from_numpy(feature_window), torch.from_numpy(target_value)


if nn is not None:

    class NHiTSBlock(nn.Module):
        def __init__(
            self,
            input_channels: int,
            lookback: int,
            hidden_size: int,
            pool_size: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.lookback = int(lookback)
            self.input_channels = int(input_channels)
            self.pool_size = max(1, int(pool_size))
            pooled_length = math.ceil(self.lookback / self.pool_size)
            pooled_input_size = pooled_length * self.input_channels
            backcast_size = self.input_channels * self.lookback

            self.mlp = nn.Sequential(
                nn.Linear(pooled_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.backcast_head = nn.Linear(hidden_size, backcast_size)
            self.forecast_head = nn.Linear(hidden_size, 1)

        def forward(self, residual):
            pooled = torch.nn.functional.avg_pool1d(
                residual,
                kernel_size=self.pool_size,
                stride=self.pool_size,
                ceil_mode=True,
            )
            hidden = self.mlp(pooled.reshape(pooled.size(0), -1))
            backcast = self.backcast_head(hidden).reshape(
                residual.size(0),
                self.input_channels,
                self.lookback,
            )
            forecast = self.forecast_head(hidden)
            return backcast, forecast


    class NHiTSModel(nn.Module):
        def __init__(
            self,
            input_channels: int,
            lookback: int,
            hidden_size: int = 256,
            pool_sizes: tuple[int, ...] = DEFAULT_POOL_SIZES,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    NHiTSBlock(
                        input_channels=input_channels,
                        lookback=lookback,
                        hidden_size=hidden_size,
                        pool_size=pool_size,
                        dropout=dropout,
                    )
                    for pool_size in pool_sizes
                ]
            )

        def forward(self, inputs):
            residual = inputs
            forecast = torch.zeros(inputs.size(0), 1, device=inputs.device, dtype=inputs.dtype)
            for block in self.blocks:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
            return forecast

else:

    class NHiTSBlock:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


    class NHiTSModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


@dataclass
class NHiTSForecaster(BaseForecaster):
    model_dir: Path | None = None
    lookback: int = 48
    hidden_size: int = 256
    pool_sizes: tuple[int, ...] = DEFAULT_POOL_SIZES
    dropout: float = 0.1
    batch_size: int = 16
    epochs: int = 15
    patience: int = 4
    learning_rate: float = 1e-3
    seed: int = 42
    model_name: str = "n_hits"
    artifact: dict[str, Any] | None = field(default=None, init=False)

    def train(self, dataframe: pd.DataFrame, zone_id: str, horizon: int) -> dict[str, Any]:
        _require_torch()
        set_deterministic_seed(self.seed)

        normalized_zone_id = normalize_zone_id(zone_id)
        feature_frame = build_zone_feature_frame(dataframe, zone_id=normalized_zone_id, dropna_lags=True)
        if feature_frame.empty:
            raise ValueError(f"Zone {normalized_zone_id} is not present in the sensor history.")

        dataset = WindowedTimeSeriesDataset(feature_frame, lookback=self.lookback, target_offset=1)
        if len(dataset) < 10:
            raise ValueError("Not enough sliding windows to train the N-HiTS forecaster.")

        train_subset, validation_subset = self._split_dataset(dataset)
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        validation_loader = DataLoader(validation_subset, batch_size=self.batch_size, shuffle=False)

        model = NHiTSModel(
            input_channels=len(NHITS_FEATURE_COLUMNS),
            lookback=self.lookback,
            hidden_size=self.hidden_size,
            pool_sizes=self.pool_sizes,
            dropout=self.dropout,
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        best_state: dict[str, Any] | None = None
        best_validation_loss = float("inf")
        stale_epochs = 0

        for _ in range(self.epochs):
            model.train()
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()

            validation_loss = self._evaluate_loss(model, validation_loader, criterion)
            if validation_loss + 1e-6 < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is None:
            raise ValueError("N-HiTS training failed to produce a valid model state.")

        model.load_state_dict(best_state)
        model.eval()

        validation_actual, validation_predicted = self._collect_predictions(model, validation_loader)
        metrics = calculate_regression_metrics(validation_actual, validation_predicted)
        residual_std = float(np.std(np.asarray(validation_actual) - np.asarray(validation_predicted), ddof=0))
        profiles = build_zone_profiles(build_zone_feature_frame(dataframe, zone_id=normalized_zone_id, dropna_lags=False))

        self.artifact = {
            "model_name": self.model_name,
            "zone_id": normalized_zone_id,
            "horizon": int(horizon),
            "lookback": int(self.lookback),
            "feature_columns": list(NHITS_FEATURE_COLUMNS),
            "hidden_size": int(self.hidden_size),
            "pool_sizes": list(self.pool_sizes),
            "dropout": float(self.dropout),
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "residual_std": residual_std,
            "profile_records": profiles.to_dict(orient="records"),
            "seed": int(self.seed),
        }

        artifact_path = self._artifact_path(normalized_zone_id, horizon)
        self.save(artifact_path)
        return {
            "artifact_path": str(artifact_path),
            "train_rows": int(len(train_subset)),
            "test_rows": int(len(validation_subset)),
            "metrics": metrics,
        }

    def predict(self, dataframe: pd.DataFrame, zone_id: str, horizon: int) -> dict[str, Any]:
        normalized_zone_id = normalize_zone_id(zone_id)
        self.load(self._artifact_path(normalized_zone_id, horizon))

        raw_history = build_zone_feature_frame(dataframe, zone_id=normalized_zone_id, dropna_lags=False)
        if raw_history.empty:
            raise ValueError(f"Zone {normalized_zone_id} is not present in the sensor history.")

        working_history = raw_history.dropna(subset=["lag_1", "lag_3", "lag_6", "lag_24"]).reset_index(drop=True)
        if len(working_history) < self.lookback:
            raise ValueError("Not enough zone history is available for N-HiTS prediction.")

        profiles = pd.DataFrame(self.artifact["profile_records"])
        model = self._build_model_from_artifact()
        residual_std = max(float(self.artifact.get("residual_std", 0.05)), 0.05)
        forecast_rows: list[dict[str, Any]] = []

        for _ in range(horizon):
            next_timestamp = pd.to_datetime(working_history["timestamp"].iloc[-1], utc=True) + pd.Timedelta(hours=1)
            exogenous = lookup_exogenous_profile(profiles, next_timestamp, working_history)
            predicted_load = max(0.0, self._predict_next_step(model, working_history))
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
            working_history = self._append_forecast_row(
                working_history=working_history,
                timestamp=next_timestamp,
                zone_id=normalized_zone_id,
                predicted_load=predicted_load,
                exogenous=exogenous,
            )

        return {
            "zone_id": normalized_zone_id,
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
            "metrics": self.artifact["metrics"],
            "model_type": self.model_name,
        }

    def evaluate(self, dataframe: pd.DataFrame, zone_id: str, horizon: int) -> dict[str, Any]:
        normalized_zone_id = normalize_zone_id(zone_id)
        feature_frame = build_zone_feature_frame(dataframe, zone_id=normalized_zone_id, dropna_lags=False)
        evaluation_frame = feature_frame.dropna(subset=["lag_1", "lag_3", "lag_6", "lag_24"]).reset_index(drop=True)
        if len(evaluation_frame) <= self.lookback + horizon:
            raise ValueError("Not enough hourly rows to evaluate the requested horizon.")

        history_frame = evaluation_frame.iloc[:-horizon].copy()
        actual_values = evaluation_frame["load_mw"].iloc[-horizon:].astype(float).tolist()
        prediction = self.predict(history_frame, normalized_zone_id, horizon)
        predicted_values = [item["predicted_load_mw"] for item in prediction["forecast_series"]]

        return {
            "zone_id": normalized_zone_id,
            "model": self.model_name,
            "horizon": horizon,
            "metrics": calculate_regression_metrics(actual_values, predicted_values),
            "actual": actual_values,
            "predicted": predicted_values,
        }

    def save(self, path: Path) -> None:
        _require_torch()
        if self.artifact is None:
            raise FileNotFoundError("No trained N-HiTS artifact is available to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.artifact, path)

    def load(self, path: Path) -> "NHiTSForecaster":
        _require_torch()
        if not path.exists():
            raise FileNotFoundError(f"Forecast artifact not found at {path}")
        self.artifact = torch.load(path, map_location="cpu")
        self.lookback = int(self.artifact.get("lookback", self.lookback))
        self.hidden_size = int(self.artifact.get("hidden_size", self.hidden_size))
        self.pool_sizes = tuple(self.artifact.get("pool_sizes", self.pool_sizes))
        self.dropout = float(self.artifact.get("dropout", self.dropout))
        self.seed = int(self.artifact.get("seed", self.seed))
        return self

    def _artifact_path(self, zone_id: str, horizon: int) -> Path:
        if self.model_dir is not None:
            directory = self.model_dir.parent / "forecasting" if self.model_dir.suffix else self.model_dir
        else:
            from app.forecasting.model_registry import model_artifact_path

            return model_artifact_path(self.model_name, zone_id, horizon, None)

        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self.model_name}_{normalize_zone_id(zone_id)}_{int(horizon)}.pt"

    def _split_dataset(self, dataset: WindowedTimeSeriesDataset) -> tuple[Subset, Subset]:
        total_windows = len(dataset)
        split_index = max(1, int(total_windows * 0.8))
        if split_index >= total_windows:
            split_index = total_windows - 1
        return (
            Subset(dataset, list(range(0, split_index))),
            Subset(dataset, list(range(split_index, total_windows))),
        )

    def _evaluate_loss(self, model, loader, criterion) -> float:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_features, batch_targets in loader:
                predictions = model(batch_features)
                losses.append(float(criterion(predictions, batch_targets).item()))
        return float(np.mean(losses)) if losses else 0.0

    def _collect_predictions(self, model, loader) -> tuple[list[float], list[float]]:
        actual: list[float] = []
        predicted: list[float] = []
        model.eval()
        with torch.no_grad():
            for batch_features, batch_targets in loader:
                outputs = model(batch_features).squeeze(-1)
                actual.extend(batch_targets.squeeze(-1).cpu().numpy().astype(float).tolist())
                predicted.extend(outputs.cpu().numpy().astype(float).tolist())
        return actual, predicted

    def _build_model_from_artifact(self):
        _require_torch()
        if self.artifact is None:
            raise FileNotFoundError("Forecast artifact is not loaded.")
        model = NHiTSModel(
            input_channels=len(NHITS_FEATURE_COLUMNS),
            lookback=int(self.artifact["lookback"]),
            hidden_size=int(self.artifact["hidden_size"]),
            pool_sizes=tuple(self.artifact["pool_sizes"]),
            dropout=float(self.artifact["dropout"]),
        )
        model.load_state_dict(self.artifact["state_dict"])
        model.eval()
        return model

    def _predict_next_step(self, model, working_history: pd.DataFrame) -> float:
        window = (
            working_history
            .iloc[-self.lookback :]
            .loc[:, NHITS_FEATURE_COLUMNS]
            .astype(float)
            .to_numpy(dtype=np.float32)
            .T
        )
        with torch.no_grad():
            tensor = torch.from_numpy(window).unsqueeze(0)
            return float(model(tensor).squeeze().item())

    def _append_forecast_row(
        self,
        working_history: pd.DataFrame,
        timestamp: pd.Timestamp,
        zone_id: str,
        predicted_load: float,
        exogenous: dict[str, float],
    ) -> pd.DataFrame:
        load_series = working_history["load_mw"].astype(float)
        next_row = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "zone_id": zone_id,
                    "load_mw": float(predicted_load),
                    "voltage_pu": exogenous["voltage_pu"],
                    "frequency_hz": exogenous["frequency_hz"],
                    "temperature_c": exogenous["temperature_c"],
                    "humidity_percent": exogenous["humidity_percent"],
                    "renewable_generation_mw": exogenous["renewable_generation_mw"],
                    "generation_mw": exogenous["generation_mw"],
                    "storage_soc_percent": exogenous["storage_soc_percent"],
                    "hour": timestamp.hour,
                    "day_of_week": timestamp.dayofweek,
                    "is_weekend": int(timestamp.dayofweek >= 5),
                    "lag_1": float(load_series.iloc[-1]),
                    "lag_3": float(load_series.iloc[-3:].mean()),
                    "lag_6": float(load_series.iloc[-6:].mean()),
                    "lag_24": float(load_series.iloc[-24] if len(load_series) >= 24 else load_series.iloc[0]),
                    "rolling_mean_3": float(load_series.iloc[-3:].mean()),
                    "rolling_mean_6": float(load_series.iloc[-6:].mean()),
                    "rolling_mean_24": float(load_series.iloc[-24:].mean()),
                }
            ]
        )
        return pd.concat([working_history, next_row], ignore_index=True)
