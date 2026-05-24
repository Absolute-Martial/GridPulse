"""Synthetic canonical 15-minute AMI/SMU history generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.forecasting.schema import (
    CANONICAL_HISTORY_COLUMNS,
    CANONICAL_INTERVAL_HOURS,
    ensure_canonical_history,
)

DEFAULT_END_TIMESTAMP: Final[pd.Timestamp] = pd.Timestamp(
    "2026-01-31T23:45:00Z"
)


@dataclass(frozen=True, slots=True)
class EntityProfile:
    entity_type: str
    entity_id: str
    secondary_substation_id: str
    transformer_id: str
    feeder_id: str
    feeder_type: str
    customer_group_id: str
    customer_type: str
    enterprise_id: str
    is_dedicated_line: int
    contracted_md_kw: float
    base_load_kw: float
    morning_peak_kw: float
    evening_peak_kw: float
    weather_sensitivity_kw: float
    variability_kw: float


ENTITY_PROFILES: Final[tuple[EntityProfile, ...]] = (
    EntityProfile(
        entity_type="substation",
        entity_id="SS_KTM_01",
        secondary_substation_id="SS_KTM_01",
        transformer_id="TR_SS_01",
        feeder_id="",
        feeder_type="mixed",
        customer_group_id="",
        customer_type="mixed",
        enterprise_id="",
        is_dedicated_line=0,
        contracted_md_kw=1600.0,
        base_load_kw=1080.0,
        morning_peak_kw=120.0,
        evening_peak_kw=210.0,
        weather_sensitivity_kw=3.0,
        variability_kw=18.0,
    ),
    EntityProfile(
        entity_type="feeder",
        entity_id="FD_RES_01",
        secondary_substation_id="SS_KTM_01",
        transformer_id="TR_FD_01",
        feeder_id="FD_RES_01",
        feeder_type="residential",
        customer_group_id="CG_RES_01",
        customer_type="residential",
        enterprise_id="",
        is_dedicated_line=0,
        contracted_md_kw=520.0,
        base_load_kw=310.0,
        morning_peak_kw=34.0,
        evening_peak_kw=82.0,
        weather_sensitivity_kw=1.2,
        variability_kw=9.0,
    ),
    EntityProfile(
        entity_type="feeder",
        entity_id="FD_COM_01",
        secondary_substation_id="SS_KTM_01",
        transformer_id="TR_FD_02",
        feeder_id="FD_COM_01",
        feeder_type="commercial",
        customer_group_id="CG_COM_01",
        customer_type="commercial",
        enterprise_id="",
        is_dedicated_line=0,
        contracted_md_kw=610.0,
        base_load_kw=355.0,
        morning_peak_kw=70.0,
        evening_peak_kw=36.0,
        weather_sensitivity_kw=1.4,
        variability_kw=11.0,
    ),
    EntityProfile(
        entity_type="customer_group",
        entity_id="CG_RES_01",
        secondary_substation_id="SS_KTM_01",
        transformer_id="TR_FD_01",
        feeder_id="FD_RES_01",
        feeder_type="residential",
        customer_group_id="CG_RES_01",
        customer_type="residential",
        enterprise_id="",
        is_dedicated_line=0,
        contracted_md_kw=180.0,
        base_load_kw=118.0,
        morning_peak_kw=14.0,
        evening_peak_kw=32.0,
        weather_sensitivity_kw=0.7,
        variability_kw=4.0,
    ),
    EntityProfile(
        entity_type="customer_group",
        entity_id="CG_IND_01",
        secondary_substation_id="SS_KTM_01",
        transformer_id="TR_FD_03",
        feeder_id="FD_IND_01",
        feeder_type="industrial",
        customer_group_id="CG_IND_01",
        customer_type="industrial",
        enterprise_id="",
        is_dedicated_line=1,
        contracted_md_kw=250.0,
        base_load_kw=172.0,
        morning_peak_kw=10.0,
        evening_peak_kw=8.0,
        weather_sensitivity_kw=0.5,
        variability_kw=3.5,
    ),
    EntityProfile(
        entity_type="enterprise",
        entity_id="ENT_CEMENT_01",
        secondary_substation_id="SS_KTM_01",
        transformer_id="TR_ENT_01",
        feeder_id="FD_IND_01",
        feeder_type="industrial",
        customer_group_id="CG_IND_01",
        customer_type="industrial",
        enterprise_id="ENT_CEMENT_01",
        is_dedicated_line=1,
        contracted_md_kw=420.0,
        base_load_kw=295.0,
        morning_peak_kw=22.0,
        evening_peak_kw=18.0,
        weather_sensitivity_kw=0.4,
        variability_kw=5.0,
    ),
    EntityProfile(
        entity_type="enterprise",
        entity_id="ENT_HOSPITAL_01",
        secondary_substation_id="SS_KTM_01",
        transformer_id="TR_ENT_02",
        feeder_id="FD_COM_01",
        feeder_type="commercial",
        customer_group_id="CG_COM_01",
        customer_type="commercial",
        enterprise_id="ENT_HOSPITAL_01",
        is_dedicated_line=1,
        contracted_md_kw=360.0,
        base_load_kw=244.0,
        morning_peak_kw=18.0,
        evening_peak_kw=26.0,
        weather_sensitivity_kw=0.6,
        variability_kw=4.5,
    ),
)


def generate_ami_history(days: int = 30, seed: int | None = None) -> pd.DataFrame:
    """Generate offline canonical 15-minute AMI/SMU history."""

    if days < 1:
        raise ValueError("days must be at least 1")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        end=DEFAULT_END_TIMESTAMP,
        periods=days * 24 * 4,
        freq="15min",
        tz="UTC",
    )
    records: list[dict[str, object]] = []

    for timestamp in timestamps:
        season, season_index = _resolve_season(timestamp.month)
        temperature_c = _ambient_temperature(timestamp, season_index, rng)
        humidity_percent = _ambient_humidity(timestamp, rng)
        is_weekend = int(timestamp.day_of_week >= 5)
        day_type = "weekend" if is_weekend else "weekday"
        slot_index = int(timestamp.hour * 4 + (timestamp.minute // 15))

        for profile in ENTITY_PROFILES:
            fingerprint_mean_kw = _fingerprint_mean(profile, slot_index, season_index, is_weekend)
            load_kw = _simulate_load(
                profile=profile,
                timestamp=timestamp,
                fingerprint_mean_kw=fingerprint_mean_kw,
                temperature_c=temperature_c,
                rng=rng,
            )
            interval_energy_kwh = load_kw * CANONICAL_INTERVAL_HOURS
            production_schedule_kw = (
                round(profile.base_load_kw * 0.08, 3)
                if profile.entity_type == "enterprise"
                else 0.0
            )

            records.append(
                {
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
                    "load_kw": round(load_kw, 3),
                    "interval_energy_kwh": round(interval_energy_kwh, 3),
                    "temperature_c": round(temperature_c, 3),
                    "humidity_percent": round(humidity_percent, 3),
                    "day_type": day_type,
                    "hour": timestamp.hour,
                    "minute": timestamp.minute,
                    "slot_index": slot_index,
                    "month": timestamp.month,
                    "day_of_week": timestamp.day_of_week,
                    "season": season,
                    "season_index": season_index,
                    "is_weekend": is_weekend,
                    "is_holiday": 0,
                    "data_quality_flag": "synthetic_ok",
                    "source_type": "synthetic_ami",
                    "production_schedule_kw": production_schedule_kw,
                    "fingerprint_mean_kw": round(fingerprint_mean_kw, 3),
                    "fingerprint_p10_kw": round(fingerprint_mean_kw * 0.92, 3),
                    "fingerprint_p90_kw": round(fingerprint_mean_kw * 1.08, 3),
                }
            )

    frame = pd.DataFrame.from_records(records, columns=CANONICAL_HISTORY_COLUMNS)
    return ensure_canonical_history(frame)


def save_ami_history(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist canonical AMI history to the local offline store."""

    output_path = path or get_settings().ami_history_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_canonical_history(frame).to_csv(output_path, index=False)
    return output_path


def load_ami_history(path: Path | None = None) -> pd.DataFrame | None:
    """Load canonical AMI history from the local offline store if present."""

    input_path = path or get_settings().ami_history_path
    if not input_path.exists():
        return None
    return ensure_canonical_history(pd.read_csv(input_path))


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
    timestamp: pd.Timestamp, season_index: int, rng: np.random.Generator
) -> float:
    season_base = {0: 11.0, 1: 19.0, 2: 28.0, 3: 24.0, 4: 18.0}[season_index]
    daily_cycle = 4.2 * np.sin(((timestamp.hour - 6) / 24) * 2 * np.pi)
    return season_base + daily_cycle + rng.normal(0.0, 0.6)


def _ambient_humidity(
    timestamp: pd.Timestamp, rng: np.random.Generator
) -> float:
    daily_cycle = 10.0 * np.cos(((timestamp.hour + 1) / 24) * 2 * np.pi)
    return float(np.clip(63.0 + daily_cycle + rng.normal(0.0, 2.2), 24.0, 96.0))


def _fingerprint_mean(
    profile: EntityProfile,
    slot_index: int,
    season_index: int,
    is_weekend: int,
) -> float:
    hour_fraction = slot_index / 4
    morning_profile = np.exp(-((hour_fraction - 9.0) ** 2) / 5.2)
    evening_profile = np.exp(-((hour_fraction - 19.0) ** 2) / 7.5)
    overnight_profile = 0.16 * np.cos((hour_fraction / 24) * 2 * np.pi)
    weekend_adjustment = -18.0 if is_weekend and profile.customer_type != "industrial" else 0.0
    season_adjustment = (season_index - 1.5) * profile.weather_sensitivity_kw * 6.0
    return max(
        8.0,
        profile.base_load_kw
        + profile.morning_peak_kw * morning_profile
        + profile.evening_peak_kw * evening_profile
        + profile.base_load_kw * overnight_profile
        + weekend_adjustment
        + season_adjustment,
    )


def _simulate_load(
    profile: EntityProfile,
    timestamp: pd.Timestamp,
    fingerprint_mean_kw: float,
    temperature_c: float,
    rng: np.random.Generator,
) -> float:
    midday_solar_offset = 0.0
    if profile.entity_type in {"substation", "feeder", "customer_group"}:
        midday_solar_offset = -0.035 * profile.base_load_kw * max(
            0.0,
            np.sin(((timestamp.hour - 6) / 12) * np.pi),
        )

    weather_adjustment = profile.weather_sensitivity_kw * max(0.0, temperature_c - 24.0)
    noise = rng.normal(0.0, profile.variability_kw)
    return max(5.0, fingerprint_mean_kw + midday_solar_offset + weather_adjustment + noise)
