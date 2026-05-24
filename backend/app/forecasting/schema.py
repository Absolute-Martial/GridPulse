"""Canonical 15-minute forecasting schema validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CANONICAL_RESOLUTION_MINUTES = 15
CANONICAL_INTERVAL_HOURS = CANONICAL_RESOLUTION_MINUTES / 60
HORIZON_TO_STEPS = {"1h": 4, "4h": 16, "24h": 96}
FINGERPRINT_COLUMNS = (
    "fingerprint_mean_kw",
    "fingerprint_p10_kw",
    "fingerprint_p90_kw",
)
CANONICAL_HISTORY_COLUMNS = (
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
) + FINGERPRINT_COLUMNS


@dataclass(slots=True)
class ForecastSchemaError(ValueError):
    """Base error for canonical forecasting schema issues."""

    detail: str

    def __str__(self) -> str:
        return self.detail


class ForecastHorizonError(ForecastSchemaError):
    """Raised when a forecast horizon is invalid."""


class ForecastEntityError(ForecastSchemaError):
    """Raised when target entity information is missing or invalid."""


class ForecastHistoryError(ForecastSchemaError):
    """Raised when canonical history is missing or malformed."""


class ForecastInsufficientHistoryError(ForecastHistoryError):
    """Raised when canonical history does not include enough rows."""


class ForecastFingerprintError(ForecastSchemaError):
    """Raised when canonical fingerprint data is missing."""


class ForecastUntrainedModelError(ForecastSchemaError):
    """Raised when forecast artifacts have not been trained yet."""


def normalize_horizon(horizon: str) -> tuple[str, int]:
    """Normalize a canonical horizon string to forecast steps."""

    if not isinstance(horizon, str):
        raise ForecastHorizonError("Forecast horizon must be a string.")

    normalized = horizon.strip().lower()
    if normalized not in HORIZON_TO_STEPS:
        allowed = ", ".join(sorted(HORIZON_TO_STEPS))
        raise ForecastHorizonError(
            f"Invalid horizon '{horizon}'. Expected one of {allowed}."
        )
    return normalized, HORIZON_TO_STEPS[normalized]


def ensure_entity_identifier(entity_type: str, entity_id: str) -> tuple[str, str]:
    """Normalize and validate a forecast target identifier."""

    if entity_type is None:
        raise ForecastEntityError("Forecast entity_type is required.")
    if entity_id is None:
        raise ForecastEntityError("Forecast entity_id is required.")

    normalized_type = str(entity_type).strip().lower()
    normalized_id = str(entity_id).strip()
    if not normalized_type:
        raise ForecastEntityError("Forecast entity_type is required.")
    if not normalized_id:
        raise ForecastEntityError("Forecast entity_id is required.")
    return normalized_type, normalized_id


def ensure_minimum_history(history: pd.DataFrame, minimum_rows: int) -> pd.DataFrame:
    """Ensure history includes enough 15-minute records for a task."""

    if len(history.index) < minimum_rows:
        raise ForecastInsufficientHistoryError(
            f"Insufficient history. Expected at least {minimum_rows} rows, found {len(history.index)}."
        )
    return history


def ensure_fingerprint_available(history: pd.DataFrame) -> pd.DataFrame:
    """Ensure the canonical fingerprint columns exist and are populated."""

    missing_columns = [column for column in FINGERPRINT_COLUMNS if column not in history.columns]
    if missing_columns:
        raise ForecastFingerprintError(
            f"Missing fingerprint columns: {', '.join(missing_columns)}."
        )

    missing_values = history.loc[:, FINGERPRINT_COLUMNS].isna().any(axis=1)
    if missing_values.any():
        raise ForecastFingerprintError("Fingerprint values are required for canonical history.")
    return history


def ensure_model_artifact_ready(artifact_path: object) -> None:
    """Ensure a model artifact exists before inference."""

    if artifact_path is None or not str(artifact_path).strip():
        raise ForecastUntrainedModelError("Forecast model artifact is not available.")
    resolved_path = Path(artifact_path).expanduser()
    if not resolved_path.exists():
        raise ForecastUntrainedModelError(
            f"Forecast model artifact does not exist: {artifact_path}."
        )
    if not resolved_path.is_file():
        raise ForecastUntrainedModelError(
            f"Forecast model artifact must be a file: {artifact_path}."
        )


def _ensure_exact_integer_series(series: pd.Series, field_name: str) -> pd.Series:
    """Validate that a metadata series contains exact integer-valued entries."""

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.isna().any():
        raise ForecastHistoryError(f"Canonical history {field_name} values must be integers.")

    fractional_values = numeric_series % 1 != 0
    if fractional_values.any():
        raise ForecastHistoryError(f"Canonical history {field_name} values must be integers.")
    return numeric_series.astype("int64")


def ensure_canonical_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize canonical 15-minute AMI history."""

    if history.empty:
        raise ForecastHistoryError("Canonical forecasting history is empty.")

    frame = history.copy()

    missing_columns = [
        column
        for column in CANONICAL_HISTORY_COLUMNS
        if column != "load_kw" and column not in frame.columns
    ]
    if missing_columns:
        fingerprint_missing = [column for column in missing_columns if column in FINGERPRINT_COLUMNS]
        if fingerprint_missing:
            raise ForecastFingerprintError(
                f"Missing fingerprint columns: {', '.join(fingerprint_missing)}."
            )
        raise ForecastHistoryError(
            f"Missing canonical history columns: {', '.join(missing_columns)}."
        )

    if "load_kw" not in frame.columns:
        frame["load_kw"] = pd.Series(float("nan"), index=frame.index, dtype="float64")
    else:
        load_kw = pd.to_numeric(frame["load_kw"], errors="coerce")
        invalid_load_values = frame["load_kw"].notna() & load_kw.isna()
        if invalid_load_values.any():
            raise ForecastHistoryError("Canonical history load_kw values must be numeric.")
        frame["load_kw"] = load_kw

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")

    for column in ("entity_type", "entity_id"):
        if frame[column].astype("string").str.strip().eq("").any():
            raise ForecastEntityError(f"Canonical history contains blank {column} values.")
        if frame[column].isna().any():
            raise ForecastEntityError(f"Canonical history contains missing {column} values.")

    load_missing = frame["load_kw"].isna()
    if load_missing.any():
        interval_energy = pd.to_numeric(frame.loc[load_missing, "interval_energy_kwh"], errors="coerce")
        if interval_energy.isna().any():
            raise ForecastHistoryError(
                "load_kw is missing and cannot be derived without interval_energy_kwh."
            )
        frame.loc[load_missing, "load_kw"] = interval_energy / CANONICAL_INTERVAL_HOURS

    minute_from_timestamp = frame["timestamp"].dt.minute
    if ((minute_from_timestamp % CANONICAL_RESOLUTION_MINUTES) != 0).any():
        raise ForecastHistoryError("Canonical history must use 15-minute AMI timestamps.")

    minute_column = _ensure_exact_integer_series(frame["minute"], "minute")
    if not (minute_column == minute_from_timestamp).all():
        raise ForecastHistoryError("Canonical history minute values must match the timestamp minute.")

    expected_slot_index = frame["timestamp"].dt.hour * 4 + (minute_from_timestamp // 15)
    slot_index = _ensure_exact_integer_series(frame["slot_index"], "slot_index")
    if not (slot_index == expected_slot_index).all():
        raise ForecastHistoryError("Canonical history slot_index values must match 15-minute timestamps.")

    ensure_fingerprint_available(frame)
    return frame.loc[:, CANONICAL_HISTORY_COLUMNS]
