import pandas as pd
import pytest

from app.core.config import get_settings
from app.forecasting.features import build_target_feature_frame
from app.forecasting.schema import (
    ForecastHorizonError,
    ForecastEntityError,
    ForecastFingerprintError,
    ForecastHistoryError,
    ForecastInsufficientHistoryError,
    ForecastUntrainedModelError,
    ensure_canonical_history,
    ensure_entity_identifier,
    ensure_model_artifact_ready,
    ensure_minimum_history,
    normalize_horizon,
)
from app.simulator.ami_history_generator import generate_ami_history


def test_normalize_horizon_accepts_canonical_strings() -> None:
    assert normalize_horizon("1h") == ("1h", 4)
    assert normalize_horizon("4h") == ("4h", 16)
    assert normalize_horizon("24h") == ("24h", 96)


def test_normalize_horizon_rejects_invalid_values() -> None:
    with pytest.raises(ForecastHorizonError):
        normalize_horizon("6")


def test_ensure_canonical_history_derives_load_kw_from_interval_energy() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-25T10:15:00Z",
                "entity_type": "feeder",
                "entity_id": "F_RES_01",
                "secondary_substation_id": "SS_01",
                "transformer_id": "TR_01",
                "feeder_id": "F_RES_01",
                "feeder_type": "residential",
                "customer_group_id": "CG_01",
                "customer_type": "residential",
                "enterprise_id": "",
                "is_dedicated_line": 0,
                "contracted_md_kw": 250.0,
                "interval_energy_kwh": 36.0,
                "temperature_c": 27.0,
                "humidity_percent": 61.0,
                "day_type": "weekday",
                "hour": 10,
                "minute": 15,
                "slot_index": 41,
                "month": 5,
                "day_of_week": 0,
                "season": "summer",
                "season_index": 2,
                "is_weekend": 0,
                "is_holiday": 0,
                "data_quality_flag": "ok",
                "source_type": "synthetic_ami",
                "production_schedule_kw": 0.0,
                "fingerprint_mean_kw": 140.0,
                "fingerprint_p10_kw": 128.0,
                "fingerprint_p90_kw": 154.0,
            }
        ]
    )

    validated = ensure_canonical_history(frame)

    assert validated.loc[0, "load_kw"] == 144.0


def test_ensure_minimum_history_raises_dedicated_exception() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-25T10:15:00Z",
                "load_kw": 144.0,
            }
        ]
    )

    with pytest.raises(ForecastInsufficientHistoryError):
        ensure_minimum_history(frame, minimum_rows=2)


def test_ensure_entity_identifier_rejects_none_entity_id() -> None:
    with pytest.raises(ForecastEntityError):
        ensure_entity_identifier("feeder", None)  # type: ignore[arg-type]


def test_ensure_model_artifact_ready_requires_existing_path(tmp_path) -> None:
    missing_path = tmp_path / "missing.pkl"

    with pytest.raises(ForecastUntrainedModelError):
        ensure_model_artifact_ready(missing_path)


def test_get_settings_reads_forecast_artifact_dir(monkeypatch, tmp_path) -> None:
    artifact_dir = tmp_path / "forecasting-artifacts"
    monkeypatch.setenv("GRIDPULSE_FORECAST_ARTIFACT_DIR", str(artifact_dir))

    settings = get_settings()

    assert settings.forecast_artifact_dir == artifact_dir.resolve()


def test_ensure_canonical_history_rejects_non_numeric_prepopulated_load_kw() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-25T10:15:00Z",
                "entity_type": "feeder",
                "entity_id": "F_RES_01",
                "secondary_substation_id": "SS_01",
                "transformer_id": "TR_01",
                "feeder_id": "F_RES_01",
                "feeder_type": "residential",
                "customer_group_id": "CG_01",
                "customer_type": "residential",
                "enterprise_id": "",
                "is_dedicated_line": 0,
                "contracted_md_kw": 250.0,
                "load_kw": "bad-value",
                "interval_energy_kwh": 36.0,
                "temperature_c": 27.0,
                "humidity_percent": 61.0,
                "day_type": "weekday",
                "hour": 10,
                "minute": 15,
                "slot_index": 41,
                "month": 5,
                "day_of_week": 0,
                "season": "summer",
                "season_index": 2,
                "is_weekend": 0,
                "is_holiday": 0,
                "data_quality_flag": "ok",
                "source_type": "synthetic_ami",
                "production_schedule_kw": 0.0,
                "fingerprint_mean_kw": 140.0,
                "fingerprint_p10_kw": 128.0,
                "fingerprint_p90_kw": 154.0,
            }
        ]
    )

    with pytest.raises(ForecastHistoryError):
        ensure_canonical_history(frame)


def test_ensure_canonical_history_rejects_minute_mismatch() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-25T10:15:00Z",
                "entity_type": "feeder",
                "entity_id": "F_RES_01",
                "secondary_substation_id": "SS_01",
                "transformer_id": "TR_01",
                "feeder_id": "F_RES_01",
                "feeder_type": "residential",
                "customer_group_id": "CG_01",
                "customer_type": "residential",
                "enterprise_id": "",
                "is_dedicated_line": 0,
                "contracted_md_kw": 250.0,
                "load_kw": 144.0,
                "interval_energy_kwh": 36.0,
                "temperature_c": 27.0,
                "humidity_percent": 61.0,
                "day_type": "weekday",
                "hour": 10,
                "minute": 30,
                "slot_index": 41,
                "month": 5,
                "day_of_week": 0,
                "season": "summer",
                "season_index": 2,
                "is_weekend": 0,
                "is_holiday": 0,
                "data_quality_flag": "ok",
                "source_type": "synthetic_ami",
                "production_schedule_kw": 0.0,
                "fingerprint_mean_kw": 140.0,
                "fingerprint_p10_kw": 128.0,
                "fingerprint_p90_kw": 154.0,
            }
        ]
    )

    with pytest.raises(ForecastHistoryError):
        ensure_canonical_history(frame)


def test_ensure_canonical_history_rejects_fractional_slot_index() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-25T10:15:00Z",
                "entity_type": "feeder",
                "entity_id": "F_RES_01",
                "secondary_substation_id": "SS_01",
                "transformer_id": "TR_01",
                "feeder_id": "F_RES_01",
                "feeder_type": "residential",
                "customer_group_id": "CG_01",
                "customer_type": "residential",
                "enterprise_id": "",
                "is_dedicated_line": 0,
                "contracted_md_kw": 250.0,
                "load_kw": 144.0,
                "interval_energy_kwh": 36.0,
                "temperature_c": 27.0,
                "humidity_percent": 61.0,
                "day_type": "weekday",
                "hour": 10,
                "minute": 15,
                "slot_index": 41.9,
                "month": 5,
                "day_of_week": 0,
                "season": "summer",
                "season_index": 2,
                "is_weekend": 0,
                "is_holiday": 0,
                "data_quality_flag": "ok",
                "source_type": "synthetic_ami",
                "production_schedule_kw": 0.0,
                "fingerprint_mean_kw": 140.0,
                "fingerprint_p10_kw": 128.0,
                "fingerprint_p90_kw": 154.0,
            }
        ]
    )

    with pytest.raises(ForecastHistoryError):
        ensure_canonical_history(frame)


def test_ensure_canonical_history_rejects_missing_fingerprint_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-25T10:15:00Z",
                "entity_type": "feeder",
                "entity_id": "F_RES_01",
                "secondary_substation_id": "SS_01",
                "transformer_id": "TR_01",
                "feeder_id": "F_RES_01",
                "feeder_type": "residential",
                "customer_group_id": "CG_01",
                "customer_type": "residential",
                "enterprise_id": "",
                "is_dedicated_line": 0,
                "contracted_md_kw": 250.0,
                "load_kw": 144.0,
                "interval_energy_kwh": 36.0,
                "temperature_c": 27.0,
                "humidity_percent": 61.0,
                "day_type": "weekday",
                "hour": 10,
                "minute": 15,
                "slot_index": 41,
                "month": 5,
                "day_of_week": 0,
                "season": "summer",
                "season_index": 2,
                "is_weekend": 0,
                "is_holiday": 0,
                "data_quality_flag": "ok",
                "source_type": "synthetic_ami",
                "production_schedule_kw": 0.0,
                "fingerprint_mean_kw": None,
                "fingerprint_p10_kw": 128.0,
                "fingerprint_p90_kw": 154.0,
            }
        ]
    )

    with pytest.raises(ForecastFingerprintError):
        ensure_canonical_history(frame)


def test_ensure_model_artifact_ready_rejects_directory(tmp_path) -> None:
    artifact_dir = tmp_path / "artifact-dir"
    artifact_dir.mkdir()

    with pytest.raises(ForecastUntrainedModelError):
        ensure_model_artifact_ready(artifact_dir)


def test_build_target_feature_frame_respects_lookback_and_has_target_column() -> None:
    history = generate_ami_history(days=7, seed=7)

    feature_frame = build_target_feature_frame(
        history,
        entity_type="feeder",
        entity_id="FD_RES_01",
        lookback_steps=480,
        horizon_steps=4,
    )

    assert len(feature_frame) >= 96
    assert {
        "lag_1",
        "lag_4",
        "lag_16",
        "lag_96",
        "rolling_mean_4",
        "rolling_mean_16",
        "rolling_mean_96",
        "target_load_kw",
    } <= set(feature_frame.columns)
    assert feature_frame["target_load_kw"].notna().all()
    assert feature_frame["entity_id"].eq("FD_RES_01").all()
