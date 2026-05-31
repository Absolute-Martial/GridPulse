"""Monte Carlo rolling AMI generation with grid-physics constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.schema import (
    CANONICAL_HISTORY_COLUMNS,
    CANONICAL_INTERVAL_HOURS,
    ensure_canonical_history,
)
from app.simulator.ami_history_generator import ENTITY_PROFILES, EntityProfile


def append_continuous_ami_history(
    history: pd.DataFrame,
    steps: int = 4,
    seed: int | None = None,
) -> pd.DataFrame:
    """Append future 15-minute AMI rows using bounded Monte Carlo variation."""

    if steps < 1:
        raise ValueError("steps must be at least 1")

    base = ensure_canonical_history(history)
    latest_timestamp = pd.to_datetime(base["timestamp"], utc=True).max()
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for offset in range(1, steps + 1):
        timestamp = latest_timestamp + pd.Timedelta(minutes=15 * offset)
        records.extend(_generate_step_records(timestamp=timestamp, rng=rng))

    generated = pd.DataFrame.from_records(records, columns=CANONICAL_HISTORY_COLUMNS)
    combined = pd.concat([base, ensure_canonical_history(generated)], ignore_index=True)
    return ensure_canonical_history(combined)


def _generate_step_records(
    timestamp: pd.Timestamp,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    season, season_index = _resolve_season(timestamp.month)
    temperature_c = _ambient_temperature(timestamp, season_index, rng)
    humidity_percent = _ambient_humidity(timestamp, season_index, rng)
    is_weekend = int(timestamp.day_of_week >= 5)
    day_type = "weekend" if is_weekend else "weekday"
    slot_index = int(timestamp.hour * 4 + timestamp.minute // 15)

    feeder_load_sum = 0.0
    provisional_records: list[dict[str, object]] = []
    substation_index: int | None = None

    for profile in ENTITY_PROFILES:
        fingerprint_mean_kw = _fingerprint_mean(profile, slot_index, season_index, is_weekend, rng)
        load_kw = _simulate_load(
            profile=profile,
            timestamp=timestamp,
            fingerprint_mean_kw=fingerprint_mean_kw,
            temperature_c=temperature_c,
            humidity_percent=humidity_percent,
            rng=rng,
        )
        if profile.entity_type == "feeder":
            feeder_load_sum += load_kw
        if profile.entity_type == "substation":
            substation_index = len(provisional_records)

        provisional_records.append(
            _record_for_profile(
                profile=profile,
                timestamp=timestamp,
                season=season,
                season_index=season_index,
                temperature_c=temperature_c,
                humidity_percent=humidity_percent,
                day_type=day_type,
                is_weekend=is_weekend,
                slot_index=slot_index,
                load_kw=load_kw,
                fingerprint_mean_kw=fingerprint_mean_kw,
            )
        )

    if substation_index is not None and feeder_load_sum > 0:
        substation_record = provisional_records[substation_index]
        min_substation_load = feeder_load_sum * float(rng.uniform(1.03, 1.12))
        if float(substation_record["load_kw"]) < min_substation_load:
            _set_record_load(substation_record, min_substation_load)

    return provisional_records


def _record_for_profile(
    *,
    profile: EntityProfile,
    timestamp: pd.Timestamp,
    season: str,
    season_index: int,
    temperature_c: float,
    humidity_percent: float,
    day_type: str,
    is_weekend: int,
    slot_index: int,
    load_kw: float,
    fingerprint_mean_kw: float,
) -> dict[str, object]:
    bounded_load_kw = min(max(load_kw, 5.0), profile.contracted_md_kw * 1.18)
    production_schedule_kw = (
        round(profile.base_load_kw * _production_schedule_factor(timestamp), 3)
        if profile.entity_type == "enterprise"
        else 0.0
    )
    return {
        "timestamp": timestamp.isoformat(),
        "entity_type": profile.entity_type,
        "entity_id": profile.entity_id,
        "secondary_substation_id": profile.secondary_substation_id,
        "transformer_id": profile.transformer_id,
        "feeder_id": profile.feeder_id,
        "feeder_type": profile.feeder_type,
        "customer_group_id": profile.customer_group_id,
        "customer_type": profile.customer_type,
        "enterprise_id": profile.enterprise_id,
        "is_dedicated_line": profile.is_dedicated_line,
        "contracted_md_kw": profile.contracted_md_kw,
        "load_kw": round(float(bounded_load_kw), 3),
        "interval_energy_kwh": round(float(bounded_load_kw) * CANONICAL_INTERVAL_HOURS, 3),
        "temperature_c": round(float(temperature_c), 3),
        "humidity_percent": round(float(humidity_percent), 3),
        "day_type": day_type,
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "slot_index": slot_index,
        "month": timestamp.month,
        "day_of_week": timestamp.day_of_week,
        "season": season,
        "season_index": season_index,
        "is_weekend": is_weekend,
        "is_holiday": int(_is_synthetic_holiday(timestamp)),
        "data_quality_flag": "synthetic_monte_carlo",
        "source_type": "continuous_synthetic_ami",
        "production_schedule_kw": production_schedule_kw,
        "fingerprint_mean_kw": round(float(fingerprint_mean_kw), 3),
        "fingerprint_p10_kw": round(float(fingerprint_mean_kw) * 0.90, 3),
        "fingerprint_p90_kw": round(float(fingerprint_mean_kw) * 1.10, 3),
    }


def _set_record_load(record: dict[str, object], load_kw: float) -> None:
    bounded = min(float(load_kw), float(record["contracted_md_kw"]) * 1.18)
    record["load_kw"] = round(bounded, 3)
    record["interval_energy_kwh"] = round(bounded * CANONICAL_INTERVAL_HOURS, 3)


def _resolve_season(month: int) -> tuple[str, int]:
    if month in (12, 1, 2):
        return "winter", 0
    if month in (3, 4):
        return "spring", 1
    if month in (5, 6):
        return "summer", 2
    if month in (7, 8, 9):
        return "monsoon", 3
    return "autumn", 4


def _ambient_temperature(
    timestamp: pd.Timestamp,
    season_index: int,
    rng: np.random.Generator,
) -> float:
    seasonal_base = {0: 11.0, 1: 20.0, 2: 29.0, 3: 25.0, 4: 18.0}[season_index]
    daily_cycle = 4.8 * np.sin(((timestamp.hour - 6) / 24) * 2 * np.pi)
    event_noise = rng.normal(0.0, 1.0)
    return float(seasonal_base + daily_cycle + event_noise)


def _ambient_humidity(
    timestamp: pd.Timestamp,
    season_index: int,
    rng: np.random.Generator,
) -> float:
    monsoon_boost = 18.0 if season_index == 3 else 0.0
    daily_cycle = 8.0 * np.cos(((timestamp.hour + 1) / 24) * 2 * np.pi)
    return float(np.clip(58.0 + monsoon_boost + daily_cycle + rng.normal(0.0, 3.0), 25.0, 96.0))


def _fingerprint_mean(
    profile: EntityProfile,
    slot_index: int,
    season_index: int,
    is_weekend: int,
    rng: np.random.Generator,
) -> float:
    hour_fraction = slot_index / 4
    morning = np.exp(-((hour_fraction - 8.5) ** 2) / 5.0)
    evening = np.exp(-((hour_fraction - 19.0) ** 2) / 7.0)
    valley = 0.14 * np.cos((hour_fraction / 24) * 2 * np.pi)
    weekend_adjustment = -0.08 * profile.base_load_kw if is_weekend and profile.customer_type != "industrial" else 0.0
    season_adjustment = (season_index - 1.5) * profile.weather_sensitivity_kw * 5.5
    monte_carlo_scale = rng.normal(1.0, 0.018)
    return max(
        6.0,
        (
            profile.base_load_kw
            + profile.morning_peak_kw * morning
            + profile.evening_peak_kw * evening
            + profile.base_load_kw * valley
            + weekend_adjustment
            + season_adjustment
        )
        * monte_carlo_scale,
    )


def _simulate_load(
    *,
    profile: EntityProfile,
    timestamp: pd.Timestamp,
    fingerprint_mean_kw: float,
    temperature_c: float,
    humidity_percent: float,
    rng: np.random.Generator,
) -> float:
    solar_offset = 0.0
    if profile.entity_type in {"substation", "feeder", "customer_group"}:
        solar_offset = -0.035 * profile.base_load_kw * max(
            0.0,
            np.sin(((timestamp.hour - 6) / 12) * np.pi),
        )

    heat_load = profile.weather_sensitivity_kw * max(0.0, temperature_c - 24.0)
    humidity_load = profile.weather_sensitivity_kw * 0.08 * max(0.0, humidity_percent - 70.0)
    production_load = (
        profile.base_load_kw * _production_schedule_factor(timestamp)
        if profile.entity_type == "enterprise"
        else 0.0
    )
    overload_event = 1.08 if rng.random() < 0.015 else 1.0
    noise = rng.normal(0.0, profile.variability_kw)
    return (
        fingerprint_mean_kw
        + solar_offset
        + heat_load
        + humidity_load
        + production_load
        + noise
    ) * overload_event


def _production_schedule_factor(timestamp: pd.Timestamp) -> float:
    if timestamp.hour < 7 or timestamp.hour > 20:
        return 0.02
    if 9 <= timestamp.hour <= 17:
        return 0.10
    return 0.06


def _is_synthetic_holiday(timestamp: pd.Timestamp) -> bool:
    return (timestamp.month, timestamp.day) in {(1, 15), (10, 15), (10, 16)}
