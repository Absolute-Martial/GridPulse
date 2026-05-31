"""Self-contained Streamlit runtime helpers for GridPulse forecasting demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.forecasting.features import (  # noqa: E402
    TARGET_MODEL_FEATURE_COLUMNS,
    build_target_feature_frame,
)
from app.forecasting.response_formatter import build_forecast_response  # noqa: E402
from app.forecasting.schema import ensure_canonical_history, normalize_horizon  # noqa: E402


@dataclass(slots=True)
class LoadedArtifact:
    """Loaded forecast artifact and its metadata."""

    path: Path
    artifact: dict[str, Any]
    metrics: dict[str, Any]

    @property
    def model(self) -> Any:
        return self.artifact["model"]

    @property
    def model_name(self) -> str:
        return str(self.artifact.get("model_name", "tree"))

    @property
    def entity_type(self) -> str:
        return str(self.artifact.get("entity_type", "feeder"))

    @property
    def entity_id(self) -> str:
        return str(self.artifact.get("entity_id", "FD_RES_01"))

    @property
    def horizon(self) -> str:
        return str(self.artifact.get("horizon", "1h"))

    @property
    def horizon_steps(self) -> int:
        return int(self.artifact.get("horizon_steps", 4))

    @property
    def feature_columns(self) -> list[str]:
        columns = self.artifact.get("feature_columns", list(TARGET_MODEL_FEATURE_COLUMNS))
        return [str(column) for column in columns]

    @property
    def lower_residual(self) -> float:
        return float(self.artifact.get("lower_residual", -0.08))

    @property
    def upper_residual(self) -> float:
        return float(self.artifact.get("upper_residual", 0.08))


def season_for_month(month: int) -> tuple[str, int]:
    if month in (12, 1, 2):
        return "winter", 0
    if month in (3, 4):
        return "spring", 1
    if month in (5, 6):
        return "summer", 2
    if month in (7, 8, 9):
        return "monsoon", 3
    return "autumn", 4


def available_model_paths() -> list[Path]:
    """Return candidate model artifacts that can be rendered in Streamlit."""

    paths: list[Path] = []
    env_override = _env_model_path()
    if env_override is not None and env_override.exists():
        paths.append(env_override)

    for directory in (
        REPO_ROOT / "data" / "models" / "forecasting",
        REPO_ROOT / "kaggle" / "output",
        REPO_ROOT / "kaggle" / "output-notebook-test",
    ):
        if directory.exists():
            paths.extend(sorted(directory.glob("*.pkl")))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and resolved.exists():
            unique_paths.append(resolved)
            seen.add(resolved)
    unique_paths.sort(key=lambda candidate: candidate.stat().st_mtime, reverse=True)
    return unique_paths


def load_artifact(path: Path | None = None) -> LoadedArtifact:
    """Load the chosen artifact and its sidecar metrics."""

    selected_path = path or _select_default_model_path()
    if selected_path is None:
        raise FileNotFoundError("No GridPulse forecast artifact was found.")
    with selected_path.open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)
    metrics_path = selected_path.with_suffix(".json")
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    elif isinstance(artifact, dict) and "metrics" in artifact:
        metrics = {"metrics": artifact["metrics"]}
    return LoadedArtifact(path=selected_path, artifact=artifact, metrics=metrics)


def history_source_paths(entity_type: str) -> list[Path]:
    normalized = entity_type.strip().lower()
    if normalized == "substation":
        preferred = [
            REPO_ROOT / "kaggle" / "dataset" / "substation.csv",
            REPO_ROOT / "data" / "forecasting" / "ami_history.csv",
        ]
    elif normalized == "enterprise":
        preferred = [
            REPO_ROOT / "data" / "forecasting" / "ami_history.csv",
            REPO_ROOT / "kaggle" / "dataset" / "feeder.csv",
        ]
    else:
        preferred = [
            REPO_ROOT / "kaggle" / "dataset" / "feeder.csv",
            REPO_ROOT / "data" / "forecasting" / "ami_history.csv",
            REPO_ROOT / "kaggle" / "dataset" / "substation.csv",
        ]
    return [path for path in preferred if path.exists()]


def load_initial_history(artifact: LoadedArtifact, lookback_steps: int = 480) -> pd.DataFrame:
    """Load or synthesize a 15-minute history for the selected artifact."""

    entity_type = artifact.entity_type
    entity_id = artifact.entity_id
    for path in history_source_paths(entity_type):
        frame = pd.read_csv(path)
        if "entity_type" in frame.columns and "entity_id" in frame.columns:
            filtered = frame[
                (frame["entity_type"].astype(str).str.lower() == entity_type.lower())
                & (frame["entity_id"].astype(str) == entity_id)
            ].copy()
            if filtered.empty:
                continue
            history = ensure_canonical_history(filtered).sort_values("timestamp").reset_index(drop=True)
            return _trim_and_validate(history, lookback_steps)

    synthesized = generate_seed_history(entity_type=entity_type, entity_id=entity_id, days=5)
    return _trim_and_validate(synthesized, lookback_steps)


def generate_seed_history(
    *,
    entity_type: str,
    entity_id: str,
    days: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a fallback canonical history that respects the grid-physics shape."""

    rng = np.random.default_rng(seed)
    periods = days * 24 * 4
    end_timestamp = pd.Timestamp.now(tz="UTC").floor("15min")
    timestamps = pd.date_range(end=end_timestamp, periods=periods, freq="15min", tz="UTC")

    if entity_type.lower() == "substation":
        profile = {
            "secondary_substation_id": "SS_KTM_01",
            "transformer_id": "TR_SS_01",
            "feeder_id": "",
            "feeder_type": "mixed",
            "customer_group_id": "",
            "customer_type": "mixed",
            "enterprise_id": "",
            "is_dedicated_line": 0,
            "contracted_md_kw": 1600.0,
            "base_load_kw": 1240.0,
            "morning_peak_kw": 120.0,
            "evening_peak_kw": 210.0,
        }
    else:
        profile = {
            "secondary_substation_id": "SS_KTM_01",
            "transformer_id": "TR_FD_01",
            "feeder_id": entity_id,
            "feeder_type": "residential",
            "customer_group_id": "CG_RES_01",
            "customer_type": "residential",
            "enterprise_id": "",
            "is_dedicated_line": 0,
            "contracted_md_kw": 520.0,
            "base_load_kw": 345.0,
            "morning_peak_kw": 34.0,
            "evening_peak_kw": 82.0,
        }

    records: list[dict[str, Any]] = []
    for timestamp in timestamps:
        slot_index = int(timestamp.hour * 4 + timestamp.minute // 15)
        day_of_week = int(timestamp.dayofweek)
        is_weekend = int(day_of_week >= 5)
        month = int(timestamp.month)
        season, season_index = season_for_month(month)
        hour = int(timestamp.hour)
        minute = int(timestamp.minute)
        hour_float = hour + minute / 60.0
        temperature_c = 6.5 + 11.0 * math.sin((slot_index / 96.0) * 2.0 * math.pi - 0.2)
        temperature_c += season_index * 1.9 + rng.normal(0.0, 0.7)
        humidity_percent = float(
            np.clip(
                70.0
                + 8.0 * math.cos((slot_index / 96.0) * 2.0 * math.pi + 0.4)
                - season_index * 2.0
                + rng.normal(0.0, 4.0),
                35.0,
                95.0,
            )
        )
        morning = profile["morning_peak_kw"] * math.exp(-0.5 * ((hour_float - 8.0) / 1.5) ** 2)
        evening = profile["evening_peak_kw"] * math.exp(-0.5 * ((hour_float - 19.0) / 2.0) ** 2)
        weekend_factor = 0.94 if is_weekend else 1.0
        fingerprint_mean_kw = (profile["base_load_kw"] + morning + evening) * weekend_factor
        weather_adjustment = max(temperature_c - 24.0, 0.0) * 1.1
        load_kw = max(
            fingerprint_mean_kw + weather_adjustment + rng.normal(0.0, profile["base_load_kw"] * 0.025),
            profile["contracted_md_kw"] * 0.35,
        )
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "entity_type": entity_type.lower(),
                "entity_id": entity_id,
                **profile,
                "load_kw": round(float(load_kw), 3),
                "interval_energy_kwh": round(float(load_kw) * 0.25, 3),
                "temperature_c": round(float(temperature_c), 3),
                "humidity_percent": round(float(humidity_percent), 3),
                "day_type": "weekend" if is_weekend else "weekday",
                "hour": hour,
                "minute": minute,
                "slot_index": slot_index,
                "month": month,
                "day_of_week": day_of_week,
                "season": season,
                "season_index": season_index,
                "is_weekend": is_weekend,
                "is_holiday": 0,
                "data_quality_flag": "synthetic_ok",
                "source_type": "synthetic_ami",
                "production_schedule_kw": 0.0,
                "fingerprint_mean_kw": round(float(fingerprint_mean_kw), 3),
                "fingerprint_p10_kw": round(float(fingerprint_mean_kw) * 0.92, 3),
                "fingerprint_p90_kw": round(float(fingerprint_mean_kw) * 1.08, 3),
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame = ensure_canonical_history(frame)
    return frame.sort_values("timestamp").reset_index(drop=True)


def append_live_step(history: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Append one synthetic 15-minute record and keep a rolling 5-day window."""

    if history.empty:
        raise ValueError("History is empty.")

    frame = ensure_canonical_history(history.sort_values("timestamp").reset_index(drop=True))
    latest = frame.iloc[-1].copy()
    next_timestamp = pd.to_datetime(latest["timestamp"], utc=True) + pd.Timedelta(minutes=15)
    rng_seed = seed if seed is not None else int(next_timestamp.value % (2**32))
    rng = np.random.default_rng(rng_seed)

    slot_index = int(next_timestamp.hour * 4 + next_timestamp.minute // 15)
    day_of_week = int(next_timestamp.dayofweek)
    is_weekend = int(day_of_week >= 5)
    month = int(next_timestamp.month)
    season, season_index = season_for_month(month)
    hour = int(next_timestamp.hour)
    minute = int(next_timestamp.minute)
    hour_float = hour + minute / 60.0

    fingerprint_group = frame[
        (frame["entity_type"].astype(str).str.lower() == str(latest["entity_type"]).lower())
        & (frame["entity_id"].astype(str) == str(latest["entity_id"]))
        & (frame["day_of_week"].astype(int) == day_of_week)
        & (frame["slot_index"].astype(int) == slot_index)
    ]
    if fingerprint_group.empty:
        fingerprint_mean_kw = float(latest["load_kw"])
        fingerprint_p10_kw = float(latest["load_kw"]) * 0.92
        fingerprint_p90_kw = float(latest["load_kw"]) * 1.08
    else:
        fingerprint_mean_kw = float(fingerprint_group["load_kw"].mean())
        fingerprint_p10_kw = float(fingerprint_group["load_kw"].quantile(0.10))
        fingerprint_p90_kw = float(fingerprint_group["load_kw"].quantile(0.90))

    temperature_c = float(
        6.5
        + 11.0 * math.sin((slot_index / 96.0) * 2.0 * math.pi - 0.2)
        + season_index * 1.9
        + rng.normal(0.0, 0.7)
    )
    humidity_percent = float(
        np.clip(
            70.0
            + 8.0 * math.cos((slot_index / 96.0) * 2.0 * math.pi + 0.4)
            - season_index * 2.0
            + rng.normal(0.0, 4.0),
            35.0,
            95.0,
        )
    )
    load_kw = max(
        fingerprint_mean_kw
        + max(temperature_c - 24.0, 0.0) * 1.1
        + (rng.normal(0.0, float(latest["contracted_md_kw"]) * 0.02))
        - (4.0 if is_weekend else 0.0),
        float(latest["contracted_md_kw"]) * 0.35,
    )

    new_row = latest.to_dict()
    new_row.update(
        {
            "timestamp": next_timestamp.isoformat(),
            "load_kw": round(float(load_kw), 3),
            "interval_energy_kwh": round(float(load_kw) * 0.25, 3),
            "temperature_c": round(float(temperature_c), 3),
            "humidity_percent": round(float(humidity_percent), 3),
            "day_type": "weekend" if is_weekend else "weekday",
            "hour": hour,
            "minute": minute,
            "slot_index": slot_index,
            "month": month,
            "day_of_week": day_of_week,
            "season": season,
            "season_index": season_index,
            "is_weekend": is_weekend,
            "is_holiday": 0,
            "data_quality_flag": "synthetic_live",
            "source_type": "synthetic_live",
            "production_schedule_kw": 0.0,
            "fingerprint_mean_kw": round(fingerprint_mean_kw, 3),
            "fingerprint_p10_kw": round(fingerprint_p10_kw, 3),
            "fingerprint_p90_kw": round(fingerprint_p90_kw, 3),
        }
    )

    updated = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
    updated = updated.tail(480).reset_index(drop=True)
    updated = _recompute_fingerprints(updated)
    return ensure_canonical_history(updated)


def build_streamlit_forecast(history: pd.DataFrame, artifact: LoadedArtifact) -> dict[str, Any]:
    """Generate a forecast response matching the backend contract."""

    normalized_horizon, _ = normalize_horizon(artifact.horizon)
    feature_frame = build_target_feature_frame(
        history,
        entity_type=artifact.entity_type,
        entity_id=artifact.entity_id,
        lookback_steps=480,
        horizon_steps=1,
    )
    if feature_frame.empty:
        raise ValueError("Not enough canonical history to build the forecast features.")

    latest_row = feature_frame.iloc[-1]
    x_latest = pd.DataFrame([latest_row.loc[artifact.feature_columns]])
    raw_prediction = artifact.model.predict(x_latest)[0]
    anchor = pd.to_datetime(latest_row["timestamp"], utc=True)

    slot_predictions: list[dict[str, Any]] = []
    for index, predicted_value in enumerate(np.atleast_1d(raw_prediction), start=1):
        timestamp = anchor + pd.Timedelta(minutes=15 * index)
        predicted_load = round(float(predicted_value), 3)
        p10 = round(float(max(predicted_value + artifact.lower_residual, 0.0)), 3)
        p90 = round(float(max(predicted_value + artifact.upper_residual, p10)), 3)
        slot_predictions.append(
            {
                "timestamp": timestamp.isoformat(),
                "predicted_load_kw": predicted_load,
                "p10_kw": p10,
                "p90_kw": p90,
                "fingerprint_mean_kw": round(float(latest_row["fingerprint_mean_kw"]), 3),
            }
        )

    confidence = "high" if len(history) >= 480 else "low"
    return build_forecast_response(
        entity_type=artifact.entity_type,
        entity_id=artifact.entity_id,
        horizon=normalized_horizon,
        latest_timestamp=anchor.isoformat(),
        slot_predictions=slot_predictions,
        confidence=confidence,
        model_name=artifact.model_name,
    )


def build_telemetry_snapshot(history: pd.DataFrame) -> dict[str, float]:
    """Compute live telemetry values from the latest canonical row."""

    if history.empty:
        raise ValueError("History is empty.")

    latest = history.sort_values("timestamp").iloc[-1]
    load_kw = float(latest["load_kw"])
    contracted = float(latest["contracted_md_kw"])
    load_ratio = load_kw / max(contracted, 1e-9)
    temperature_c = float(latest["temperature_c"])
    humidity_percent = float(latest["humidity_percent"])
    hour = int(latest["hour"])
    daylight_factor = max(math.sin((hour - 6) / 24.0 * math.pi), 0.0)
    voltage_rng = np.random.default_rng(int(load_kw * 1000) % (2**32))
    frequency_rng = np.random.default_rng(int(load_kw * 2000) % (2**32))
    storage_rng = np.random.default_rng(int(load_kw * 3000) % (2**32))
    return {
        "load_kw": round(load_kw, 3),
        "temperature_c": round(temperature_c, 3),
        "humidity_percent": round(humidity_percent, 3),
        "voltage_pu": round(
            float(
                np.clip(
                    1.02 - 0.00035 * (load_ratio * 100.0) + voltage_rng.normal(0.0, 0.004),
                    0.94,
                    1.05,
                )
            ),
            4,
        ),
        "frequency_hz": round(
            float(
                np.clip(
                    50.0 - max(load_ratio - 0.82, 0.0) * 0.02 + frequency_rng.normal(0.0, 0.012),
                    49.8,
                    50.2,
                )
            ),
            4,
        ),
        "renewable_generation_kw": round(float(load_kw * 0.05 * daylight_factor), 3),
        "storage_soc_percent": round(
            float(
                np.clip(
                    72.0 - max(load_ratio - 0.85, 0.0) * 8.0 + storage_rng.normal(0.0, 0.8),
                    20.0,
                    95.0,
                )
            ),
            3,
        ),
        "fingerprint_mean_kw": round(float(latest["fingerprint_mean_kw"]), 3),
        "fingerprint_p10_kw": round(float(latest["fingerprint_p10_kw"]), 3),
        "fingerprint_p90_kw": round(float(latest["fingerprint_p90_kw"]), 3),
    }


def build_explanation(history: pd.DataFrame, forecast: dict[str, Any]) -> str:
    """Produce a plain-language explanation for the forecast."""

    slot_predictions = forecast.get("slot_predictions", [])
    if not slot_predictions:
        return "No forecast slots were generated."

    peak_slot = max(slot_predictions, key=lambda item: float(item["predicted_load_kw"]))
    peak_time = pd.to_datetime(peak_slot["timestamp"], utc=True).strftime("%H:%M")
    latest = history.sort_values("timestamp").iloc[-1]
    latest_load = float(latest["load_kw"])
    fingerprint = float(latest["fingerprint_mean_kw"])
    temp = float(latest["temperature_c"])
    temp_trend = "above" if temp > 20.0 else "near or below"
    baseline_gap = latest_load - fingerprint
    gap_phrase = "higher" if baseline_gap >= 0 else "lower"
    return (
        f"{forecast['entity_type'].title()} {forecast['entity_id']} is forecast to peak at {peak_time} "
        f"because the latest observed load is {abs(baseline_gap):.1f} kW {gap_phrase} than the fingerprint baseline, "
        f"and temperature is {temp_trend} normal for this season."
    )


def format_metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a metrics object into display rows."""

    flattened = metrics.get("metrics", metrics)
    return [{"metric": key, "value": value} for key, value in flattened.items() if key != "job"]


def format_artifact_label(path: Path, artifact: LoadedArtifact) -> str:
    return f"{artifact.model_name} | {artifact.entity_type} {artifact.entity_id} | {artifact.horizon} | {path.name}"


def _select_default_model_path() -> Path | None:
    if (override := _env_model_path()) is not None and override.exists():
        return override
    paths = available_model_paths()
    return paths[0] if paths else None


def _env_model_path() -> Path | None:
    value = os.environ.get("GRIDPULSE_STREAMLIT_MODEL_PATH")
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        candidates = sorted(path.glob("*.pkl"))
        return candidates[0] if candidates else None
    return path


def _trim_and_validate(history: pd.DataFrame, lookback_steps: int) -> pd.DataFrame:
    trimmed = history.sort_values("timestamp").tail(lookback_steps).reset_index(drop=True)
    return ensure_canonical_history(trimmed)


def _recompute_fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    fingerprint = (
        frame.groupby(["entity_type", "entity_id", "day_of_week", "slot_index"], as_index=False)
        .agg(
            fingerprint_mean_kw=("load_kw", "mean"),
            fingerprint_p10_kw=("load_kw", lambda series: float(series.quantile(0.10))),
            fingerprint_p90_kw=("load_kw", lambda series: float(series.quantile(0.90))),
        )
        .reset_index(drop=True)
    )
    merged = frame.drop(
        columns=["fingerprint_mean_kw", "fingerprint_p10_kw", "fingerprint_p90_kw"],
        errors="ignore",
    ).merge(
        fingerprint,
        on=["entity_type", "entity_id", "day_of_week", "slot_index"],
        how="left",
    )
    merged["fingerprint_mean_kw"] = merged["fingerprint_mean_kw"].fillna(merged["load_kw"])
    merged["fingerprint_p10_kw"] = merged["fingerprint_p10_kw"].fillna(merged["load_kw"] * 0.92)
    merged["fingerprint_p90_kw"] = merged["fingerprint_p90_kw"].fillna(merged["load_kw"] * 1.08)
    return merged
