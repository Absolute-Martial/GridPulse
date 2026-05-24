"""CSV adapter for synthetic telemetry input."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.simulator.sensor_generator import REQUIRED_COLUMNS


class CsvSchemaError(ValueError):
    """Raised when the synthetic CSV source is missing required columns."""


def get_latest_sensor_record() -> dict[str, object] | None:
    frame = load_sensor_frame()
    if frame is None:
        return None

    latest = frame.sort_values("timestamp").iloc[-1]
    return _row_to_record(latest)


def get_sensor_history(hours: int) -> list[dict[str, object]]:
    frame = load_sensor_frame()
    if frame is None:
        return []

    sorted_frame = frame.sort_values("timestamp").copy()
    sorted_frame["timestamp"] = pd.to_datetime(sorted_frame["timestamp"], utc=True)
    cutoff = sorted_frame["timestamp"].max() - pd.Timedelta(hours=hours)
    history_frame = sorted_frame[sorted_frame["timestamp"] >= cutoff]
    return [_row_to_record(row) for _, row in history_frame.iterrows()]


def load_sensor_frame() -> pd.DataFrame | None:
    csv_path = get_settings().synthetic_csv_path
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    frame = pd.read_csv(csv_path)
    if frame.empty:
        return None

    _validate_columns(frame.columns, csv_path)
    return frame


def _validate_columns(columns: pd.Index, csv_path: Path) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        raise CsvSchemaError(
            f"Invalid synthetic CSV schema at {csv_path}: missing columns {missing_columns}"
        )


def _row_to_record(row: pd.Series) -> dict[str, object]:
    return {
        "timestamp": str(row["timestamp"]),
        "sensor_id": str(row["sensor_id"]),
        "zone_id": str(row["zone_id"]),
        "bus_id": str(row["bus_id"]),
        "voltage_pu": float(row["voltage_pu"]),
        "current_a": float(row["current_a"]),
        "frequency_hz": float(row["frequency_hz"]),
        "load_mw": float(row["load_mw"]),
        "generation_mw": float(row["generation_mw"]),
        "renewable_generation_mw": float(row["renewable_generation_mw"]),
        "line_id": str(row["line_id"]),
        "line_flow_mw": float(row["line_flow_mw"]),
        "line_capacity_mw": float(row["line_capacity_mw"]),
        "temperature_c": float(row["temperature_c"]),
        "humidity_percent": float(row["humidity_percent"]),
        "storage_soc_percent": float(row["storage_soc_percent"]),
        "fault_status": str(row["fault_status"]),
        "anomaly_label": str(row["anomaly_label"]),
    }
