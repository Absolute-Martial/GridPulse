"""Synthetic IoT sensor generator for the GridPulse prototype."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from app.core.config import get_settings

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "sensor_id",
    "zone_id",
    "bus_id",
    "voltage_pu",
    "current_a",
    "frequency_hz",
    "load_mw",
    "generation_mw",
    "renewable_generation_mw",
    "line_id",
    "line_flow_mw",
    "line_capacity_mw",
    "temperature_c",
    "humidity_percent",
    "storage_soc_percent",
    "fault_status",
    "anomaly_label",
)

ZONE_IDS: Final[tuple[str, ...]] = (
    "zone-1",
    "zone-2",
    "zone-3",
    "zone-4",
    "zone-5",
)

BUS_IDS: Final[tuple[str, ...]] = (
    "bus-1",
    "bus-2",
    "bus-3",
    "bus-4",
    "bus-5",
    "bus-6",
    "bus-7",
    "bus-8",
)

LINE_IDS: Final[tuple[str, ...]] = (
    "line-1",
    "line-2",
    "line-3",
    "line-4",
    "line-5",
    "line-6",
    "line-7",
    "line-8",
    "line-9",
    "line-10",
)

GENERATOR_IDS: Final[tuple[str, ...]] = ("gen-1", "gen-2")
SOLAR_ID: Final[str] = "solar-1"
WIND_ID: Final[str] = "wind-1"
STORAGE_ID: Final[str] = "battery-1"


@dataclass(frozen=True)
class LineProfile:
    line_id: str
    zone_id: str
    bus_id: str
    line_capacity_mw: float
    load_scale: float
    renewable_bias: float


LINE_PROFILES: Final[tuple[LineProfile, ...]] = (
    LineProfile("line-1", "zone-1", "bus-1", 22.0, 0.88, 0.20),
    LineProfile("line-2", "zone-1", "bus-2", 24.0, 0.94, 0.25),
    LineProfile("line-3", "zone-2", "bus-3", 26.0, 1.02, 0.85),
    LineProfile("line-4", "zone-2", "bus-4", 28.0, 0.98, 0.30),
    LineProfile("line-5", "zone-3", "bus-5", 30.0, 1.05, 0.35),
    LineProfile("line-6", "zone-3", "bus-6", 32.0, 1.00, 0.95),
    LineProfile("line-7", "zone-4", "bus-7", 24.0, 0.80, 0.25),
    LineProfile("line-8", "zone-4", "bus-8", 27.0, 0.84, 0.15),
    LineProfile("line-9", "zone-5", "bus-1", 29.0, 0.92, 0.18),
    LineProfile("line-10", "zone-5", "bus-6", 31.0, 0.97, 0.65),
)

ZONE_BASE_LOAD_MW: Final[dict[str, float]] = {
    "zone-1": 6.0,
    "zone-2": 6.8,
    "zone-3": 7.6,
    "zone-4": 5.2,
    "zone-5": 5.8,
}

GENERATOR_CAPACITY_MW: Final[dict[str, float]] = {
    "bus-1": 9.0,
    "bus-7": 8.5,
}


def generate_sensor_data(
    hours: int = 24,
    interval_minutes: int = 5,
    end_time: datetime | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = _build_timestamps(hours=hours, interval_minutes=interval_minutes, end_time=end_time)
    event_windows = _event_windows(len(timestamps))
    records: list[dict[str, object]] = []

    for timestamp_index, timestamp in enumerate(timestamps):
        hour_of_day = timestamp.hour + (timestamp.minute / 60)
        temperature_c = _ambient_temperature(hour_of_day, rng)
        humidity_percent = _ambient_humidity(hour_of_day, rng)
        solar_mw = _solar_generation(hour_of_day, rng)
        wind_mw = _wind_generation(hour_of_day, rng)
        storage_soc_percent = _storage_soc(hour_of_day, rng)

        for line_index, profile in enumerate(LINE_PROFILES):
            zone_base = ZONE_BASE_LOAD_MW[profile.zone_id]
            load_mw = max(
                0.0,
                zone_base * _daily_load_multiplier(hour_of_day) * profile.load_scale
                + rng.normal(0.0, 0.18),
            )
            renewable_generation_mw = max(
                0.0,
                solar_mw * profile.renewable_bias * 0.45
                + wind_mw * (0.15 + 0.55 * profile.renewable_bias)
                + rng.normal(0.0, 0.08),
            )
            conventional_generation_mw = _conventional_generation(
                bus_id=profile.bus_id,
                load_mw=load_mw,
                hour_of_day=hour_of_day,
                rng=rng,
            )
            generation_mw = conventional_generation_mw + renewable_generation_mw
            line_flow_mw = max(
                0.0,
                load_mw * (0.84 + (0.03 * (line_index % 3)))
                - generation_mw * 0.34
                + rng.normal(0.0, 0.25),
            )
            voltage_pu = np.clip(
                1.01
                - 0.010 * (load_mw / max(profile.line_capacity_mw, 1.0))
                + 0.006 * (generation_mw / max(profile.line_capacity_mw, 1.0))
                + rng.normal(0.0, 0.0045),
                0.90,
                1.08,
            )
            frequency_hz = np.clip(
                50.0
                + 0.018 * ((generation_mw - load_mw) / max(zone_base, 1.0))
                + rng.normal(0.0, 0.015),
                49.85,
                50.15,
            )
            current_a = max(
                0.0,
                (line_flow_mw * 1_000_000) / (np.sqrt(3) * 33_000) + rng.normal(0.0, 6.0),
            )
            anomaly_label = "normal"
            fault_status = "normal"

            if timestamp_index == event_windows["voltage_anomaly"] and line_index == 1:
                voltage_pu = 0.89
                anomaly_label = "voltage_anomaly"
            elif timestamp_index == event_windows["frequency_anomaly"] and line_index == 4:
                frequency_hz = 47.6
                anomaly_label = "frequency_anomaly"
            elif timestamp_index == event_windows["line_overload"] and line_index == 7:
                line_flow_mw = min(profile.line_capacity_mw * 1.08, 45.0)
                current_a = (line_flow_mw * 1_000_000) / (np.sqrt(3) * 33_000)
                anomaly_label = "line_overload"
            elif timestamp_index == event_windows["fault_event"] and line_index == 9:
                line_flow_mw = 0.0
                current_a = 0.0
                voltage_pu = 0.91
                fault_status = "fault"
                anomaly_label = "fault_event"

            records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "sensor_id": f"sensor-{profile.line_id}-{timestamp.strftime('%Y%m%d%H%M')}",
                    "zone_id": profile.zone_id,
                    "bus_id": profile.bus_id,
                    "voltage_pu": round(float(voltage_pu), 4),
                    "current_a": round(float(current_a), 2),
                    "frequency_hz": round(float(frequency_hz), 4),
                    "load_mw": round(float(load_mw), 3),
                    "generation_mw": round(float(generation_mw), 3),
                    "renewable_generation_mw": round(float(renewable_generation_mw), 3),
                    "line_id": profile.line_id,
                    "line_flow_mw": round(float(line_flow_mw), 3),
                    "line_capacity_mw": profile.line_capacity_mw,
                    "temperature_c": round(float(temperature_c), 2),
                    "humidity_percent": round(float(humidity_percent), 2),
                    "storage_soc_percent": round(float(storage_soc_percent), 2),
                    "fault_status": fault_status,
                    "anomaly_label": anomaly_label,
                }
            )

    return pd.DataFrame.from_records(records, columns=REQUIRED_COLUMNS)


def save_sensor_data(frame: pd.DataFrame, csv_path: Path | None = None) -> Path:
    output_path = csv_path or get_settings().synthetic_csv_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def generate_and_save_sensor_data(
    hours: int = 24,
    interval_minutes: int = 5,
    seed: int = 42,
    end_time: datetime | None = None,
) -> tuple[pd.DataFrame, Path]:
    frame = generate_sensor_data(
        hours=hours,
        interval_minutes=interval_minutes,
        end_time=end_time,
        seed=seed,
    )
    output_path = save_sensor_data(frame)
    return frame, output_path


def _build_timestamps(
    hours: int,
    interval_minutes: int,
    end_time: datetime | None,
) -> pd.DatetimeIndex:
    if hours < 1:
        raise ValueError("hours must be at least 1")
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")

    steps = max(1, int((hours * 60) / interval_minutes))
    if end_time is None:
        current_time = datetime.now(UTC).replace(second=0, microsecond=0)
        aligned_minute = current_time.minute - (current_time.minute % interval_minutes)
        end_time = current_time.replace(minute=aligned_minute)
    elif end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=UTC)

    return pd.date_range(
        end=end_time,
        periods=steps,
        freq=f"{interval_minutes}min",
        tz=UTC,
    )


def _event_windows(total_steps: int) -> dict[str, int]:
    return {
        "voltage_anomaly": max(1, total_steps // 5),
        "frequency_anomaly": max(2, total_steps // 2),
        "line_overload": max(3, total_steps - 1),
        "fault_event": max(4, total_steps - 1),
    }


def _daily_load_multiplier(hour_of_day: float) -> float:
    morning_ramp = 0.18 * np.exp(-((hour_of_day - 8.0) ** 2) / 10.0)
    evening_peak = 0.42 * np.exp(-((hour_of_day - 19.0) ** 2) / 8.0)
    overnight_dip = 0.12 * np.exp(-((hour_of_day - 3.0) ** 2) / 6.0)
    daytime_base = 0.72 + 0.14 * np.sin(((hour_of_day - 6.0) / 24.0) * 2.0 * np.pi)
    return max(0.42, daytime_base + morning_ramp + evening_peak - overnight_dip)


def _solar_generation(hour_of_day: float, rng: np.random.Generator) -> float:
    daylight = np.sin(np.pi * np.clip((hour_of_day - 6.0) / 12.0, 0.0, 1.0))
    return max(0.0, 4.8 * (daylight**1.7) + rng.normal(0.0, 0.12))


def _wind_generation(hour_of_day: float, rng: np.random.Generator) -> float:
    baseline = 1.9 + 0.65 * np.sin(((hour_of_day + 2.0) / 24.0) * 2.0 * np.pi)
    gust = 0.35 * np.sin(((hour_of_day * 3.0) / 24.0) * 2.0 * np.pi)
    return max(0.2, baseline + gust + rng.normal(0.0, 0.18))


def _conventional_generation(
    bus_id: str,
    load_mw: float,
    hour_of_day: float,
    rng: np.random.Generator,
) -> float:
    if bus_id not in GENERATOR_CAPACITY_MW:
        return 0.0

    dispatch_bias = 0.65 if hour_of_day < 17 else 0.82
    output = load_mw * dispatch_bias + rng.normal(0.0, 0.16)
    return float(np.clip(output, 0.0, GENERATOR_CAPACITY_MW[bus_id]))


def _storage_soc(hour_of_day: float, rng: np.random.Generator) -> float:
    baseline = 57.0 + 21.0 * np.sin(((hour_of_day - 8.0) / 24.0) * 2.0 * np.pi)
    return float(np.clip(baseline + rng.normal(0.0, 2.0), 8.0, 98.0))


def _ambient_temperature(hour_of_day: float, rng: np.random.Generator) -> float:
    baseline = 23.0 + 7.5 * np.sin(((hour_of_day - 8.0) / 24.0) * 2.0 * np.pi)
    return float(np.clip(baseline + rng.normal(0.0, 0.55), 8.0, 40.0))


def _ambient_humidity(hour_of_day: float, rng: np.random.Generator) -> float:
    baseline = 58.0 - 14.0 * np.sin(((hour_of_day - 8.0) / 24.0) * 2.0 * np.pi)
    return float(np.clip(baseline + rng.normal(0.0, 2.8), 18.0, 92.0))
