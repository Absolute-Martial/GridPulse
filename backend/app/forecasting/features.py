"""Feature engineering for both legacy zone and new 15-minute target forecasting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.forecasting.schema import (
    ensure_canonical_history,
    ensure_entity_identifier,
    ensure_minimum_history,
)

FORECAST_INPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "zone_id",
    "load_mw",
    "voltage_pu",
    "frequency_hz",
    "temperature_c",
    "humidity_percent",
    "renewable_generation_mw",
    "generation_mw",
    "storage_soc_percent",
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_3",
    "lag_6",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",
)

MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "voltage_pu",
    "frequency_hz",
    "temperature_c",
    "humidity_percent",
    "renewable_generation_mw",
    "generation_mw",
    "storage_soc_percent",
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_3",
    "lag_6",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",
)

TARGET_FORECAST_INPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "entity_type",
    "entity_id",
    "secondary_substation_id",
    "transformer_id",
    "feeder_id",
    "feeder_type",
    "customer_group_id",
    "customer_type",
    "enterprise_id",
    "is_dedicated_line",
    "contracted_md_kw",
    "load_kw",
    "interval_energy_kwh",
    "temperature_c",
    "humidity_percent",
    "day_type",
    "hour",
    "minute",
    "slot_index",
    "month",
    "day_of_week",
    "season",
    "season_index",
    "is_weekend",
    "is_holiday",
    "data_quality_flag",
    "source_type",
    "production_schedule_kw",
    "fingerprint_mean_kw",
    "fingerprint_p10_kw",
    "fingerprint_p90_kw",
    "lag_1",
    "lag_4",
    "lag_16",
    "lag_96",
    "rolling_mean_4",
    "rolling_mean_16",
    "rolling_mean_96",
    "target_load_kw",
)

TARGET_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "contracted_md_kw",
    "temperature_c",
    "humidity_percent",
    "hour",
    "minute",
    "slot_index",
    "day_of_week",
    "month",
    "season_index",
    "is_weekend",
    "is_holiday",
    "production_schedule_kw",
    "fingerprint_mean_kw",
    "fingerprint_p10_kw",
    "fingerprint_p90_kw",
    "lag_1",
    "lag_4",
    "lag_16",
    "lag_96",
    "rolling_mean_4",
    "rolling_mean_16",
    "rolling_mean_96",
)


def normalize_zone_id(zone_id: str) -> str:
    normalized = zone_id.strip().lower()
    if normalized.startswith("zone-"):
        return normalized
    if normalized.startswith("z") and normalized[1:].isdigit():
        return f"zone-{int(normalized[1:])}"
    if normalized.isdigit():
        return f"zone-{int(normalized)}"
    return normalized


def filter_target_history(
    history: pd.DataFrame,
    entity_type: str,
    entity_id: str,
) -> pd.DataFrame:
    """Return canonical 15-minute history for one target entity."""

    normalized_type, normalized_id = ensure_entity_identifier(entity_type, entity_id)
    frame = ensure_canonical_history(history)
    filtered = frame[
        (frame["entity_type"].astype(str).str.lower() == normalized_type)
        & (frame["entity_id"].astype(str) == normalized_id)
    ].copy()
    ensure_minimum_history(filtered, 1)
    return filtered.sort_values("timestamp").reset_index(drop=True)


def build_target_feature_frame(
    history: pd.DataFrame,
    entity_type: str,
    entity_id: str,
    lookback_steps: int = 480,
    horizon_steps: int = 1,
    dropna_lags: bool = True,
) -> pd.DataFrame:
    """Build leakage-safe 15-minute features for a single forecast target."""

    if lookback_steps < 1:
        raise ValueError("lookback_steps must be at least 1")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be at least 1")

    target = filter_target_history(history, entity_type=entity_type, entity_id=entity_id)
    working = target.copy()
    shifted_load = working["load_kw"].shift(1)
    working["lag_1"] = working["load_kw"].shift(1)
    working["lag_4"] = working["load_kw"].shift(4)
    working["lag_16"] = working["load_kw"].shift(16)
    working["lag_96"] = working["load_kw"].shift(96)
    working["rolling_mean_4"] = shifted_load.rolling(4, min_periods=1).mean()
    working["rolling_mean_16"] = shifted_load.rolling(16, min_periods=1).mean()
    working["rolling_mean_96"] = shifted_load.rolling(96, min_periods=1).mean()
    working["target_load_kw"] = working["load_kw"].shift(-horizon_steps)

    if dropna_lags:
        working = working.dropna(
            subset=[
                "lag_1",
                "lag_4",
                "lag_16",
                "lag_96",
                "target_load_kw",
            ]
        ).reset_index(drop=True)

    return working.tail(lookback_steps).loc[:, TARGET_FORECAST_INPUT_COLUMNS].reset_index(drop=True)


def build_zone_feature_frame(
    sensor_frame: pd.DataFrame,
    zone_id: str | None = None,
    dropna_lags: bool = False,
) -> pd.DataFrame:
    frame = sensor_frame.copy()
    if frame.empty:
        raise ValueError("Sensor history is empty.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["timestamp_hour"] = frame["timestamp"].dt.floor("h")
    aggregated = (
        frame.groupby(["timestamp_hour", "zone_id"], as_index=False)
        .agg(
            load_mw=("load_mw", "sum"),
            voltage_pu=("voltage_pu", "mean"),
            frequency_hz=("frequency_hz", "mean"),
            temperature_c=("temperature_c", "mean"),
            humidity_percent=("humidity_percent", "mean"),
            renewable_generation_mw=("renewable_generation_mw", "sum"),
            generation_mw=("generation_mw", "sum"),
            storage_soc_percent=("storage_soc_percent", "mean"),
        )
        .sort_values(["zone_id", "timestamp_hour"])
        .reset_index(drop=True)
    )

    aggregated["timestamp"] = aggregated["timestamp_hour"]
    aggregated["hour"] = aggregated["timestamp"].dt.hour
    aggregated["day_of_week"] = aggregated["timestamp"].dt.dayofweek
    aggregated["is_weekend"] = aggregated["day_of_week"].isin([5, 6]).astype(int)

    feature_frames: list[pd.DataFrame] = []
    for current_zone_id, zone_frame in aggregated.groupby("zone_id", sort=True):
        working = zone_frame.sort_values("timestamp").reset_index(drop=True).copy()
        working["lag_1"] = working["load_mw"].shift(1)
        working["lag_3"] = working["load_mw"].shift(3)
        working["lag_6"] = working["load_mw"].shift(6)
        working["lag_24"] = working["load_mw"].shift(24)
        shifted = working["load_mw"].shift(1)
        working["rolling_mean_3"] = shifted.rolling(3, min_periods=1).mean()
        working["rolling_mean_6"] = shifted.rolling(6, min_periods=1).mean()
        working["rolling_mean_24"] = shifted.rolling(24, min_periods=1).mean()
        feature_frames.append(working)

    feature_frame = pd.concat(feature_frames, ignore_index=True)
    if zone_id is not None:
        normalized_zone_id = normalize_zone_id(zone_id)
        feature_frame = feature_frame[feature_frame["zone_id"] == normalized_zone_id].copy()

    if dropna_lags:
        feature_frame = feature_frame.dropna(
            subset=["lag_1", "lag_3", "lag_6", "lag_24"]
        ).reset_index(drop=True)

    return feature_frame.loc[:, FORECAST_INPUT_COLUMNS].reset_index(drop=True)


def build_zone_profiles(feature_frame: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        feature_frame.groupby(["zone_id", "hour", "day_of_week"], as_index=False)
        .agg(
            voltage_pu=("voltage_pu", "mean"),
            frequency_hz=("frequency_hz", "mean"),
            temperature_c=("temperature_c", "mean"),
            humidity_percent=("humidity_percent", "mean"),
            renewable_generation_mw=("renewable_generation_mw", "mean"),
            generation_mw=("generation_mw", "mean"),
            storage_soc_percent=("storage_soc_percent", "mean"),
        )
        .sort_values(["zone_id", "day_of_week", "hour"])
        .reset_index(drop=True)
    )
    return profiles


def lookup_exogenous_profile(
    profiles: pd.DataFrame,
    timestamp: pd.Timestamp,
    zone_history: pd.DataFrame,
) -> dict[str, float]:
    matched = profiles[
        (profiles["hour"] == timestamp.hour)
        & (profiles["day_of_week"] == timestamp.dayofweek)
    ]
    if not matched.empty:
        row = matched.iloc[0]
        return {
            "voltage_pu": float(row["voltage_pu"]),
            "frequency_hz": float(row["frequency_hz"]),
            "temperature_c": float(row["temperature_c"]),
            "humidity_percent": float(row["humidity_percent"]),
            "renewable_generation_mw": float(row["renewable_generation_mw"]),
            "generation_mw": float(row["generation_mw"]),
            "storage_soc_percent": float(row["storage_soc_percent"]),
        }

    fallback = zone_history.iloc[-24:] if len(zone_history) >= 24 else zone_history
    return {
        "voltage_pu": float(fallback["voltage_pu"].mean()),
        "frequency_hz": float(fallback["frequency_hz"].mean()),
        "temperature_c": float(fallback["temperature_c"].mean()),
        "humidity_percent": float(fallback["humidity_percent"].mean()),
        "renewable_generation_mw": float(fallback["renewable_generation_mw"].mean()),
        "generation_mw": float(fallback["generation_mw"].mean()),
        "storage_soc_percent": float(fallback["storage_soc_percent"].mean()),
    }


def build_recursive_feature_row(
    zone_history: pd.DataFrame,
    timestamp: pd.Timestamp,
    exogenous: dict[str, float],
) -> pd.DataFrame:
    load_series = zone_history["load_mw"].astype(float)
    return pd.DataFrame(
        [
            {
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


def append_prediction_row(
    zone_history: pd.DataFrame,
    timestamp: pd.Timestamp,
    zone_id: str,
    predicted_load: float,
    exogenous: dict[str, float],
) -> pd.DataFrame:
    next_row = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "zone_id": zone_id,
                "load_mw": predicted_load,
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
                "lag_1": predicted_load,
                "lag_3": None,
                "lag_6": None,
                "lag_24": None,
                "rolling_mean_3": None,
                "rolling_mean_6": None,
                "rolling_mean_24": None,
            }
        ]
    )
    return pd.concat([zone_history, next_row], ignore_index=True)
