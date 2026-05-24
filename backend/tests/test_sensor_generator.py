from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.simulator.sensor_generator import (
    REQUIRED_COLUMNS,
    generate_sensor_data,
)


def test_generate_sensor_data_has_required_columns_and_topology() -> None:
    frame = generate_sensor_data(
        hours=24,
        interval_minutes=60,
        end_time=datetime(2026, 5, 24, 23, 0, tzinfo=UTC),
        seed=7,
    )

    assert list(frame.columns) == list(REQUIRED_COLUMNS)
    assert not frame.empty
    assert frame["zone_id"].nunique() == 5
    assert frame["bus_id"].nunique() == 8
    assert frame["line_id"].nunique() == 10
    assert frame["sensor_id"].is_unique


def test_generate_sensor_data_stays_within_expected_value_ranges() -> None:
    frame = generate_sensor_data(
        hours=24,
        interval_minutes=60,
        end_time=datetime(2026, 5, 24, 23, 0, tzinfo=UTC),
        seed=7,
    )

    assert frame["voltage_pu"].between(0.88, 1.12).all()
    assert frame["current_a"].between(0.0, 1200.0).all()
    assert frame["frequency_hz"].between(47.0, 53.0).all()
    assert frame["load_mw"].between(0.0, 25.0).all()
    assert frame["generation_mw"].between(0.0, 30.0).all()
    assert frame["renewable_generation_mw"].between(0.0, 12.0).all()
    assert frame["line_flow_mw"].between(0.0, 45.0).all()
    assert frame["line_capacity_mw"].between(15.0, 40.0).all()
    assert frame["temperature_c"].between(5.0, 45.0).all()
    assert frame["humidity_percent"].between(10.0, 95.0).all()
    assert frame["storage_soc_percent"].between(5.0, 100.0).all()
    assert set(frame["fault_status"].unique()) <= {"normal", "fault"}
    assert set(frame["anomaly_label"].unique()) <= {
        "normal",
        "voltage_anomaly",
        "frequency_anomaly",
        "line_overload",
        "fault_event",
    }


def test_generate_sensor_data_models_daily_patterns_and_rare_events() -> None:
    frame = generate_sensor_data(
        hours=24,
        interval_minutes=60,
        end_time=datetime(2026, 5, 24, 23, 0, tzinfo=UTC),
        seed=7,
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    by_hour = frame.groupby(frame["timestamp"].dt.hour).mean(numeric_only=True)

    assert by_hour.loc[19, "load_mw"] > by_hour.loc[3, "load_mw"]
    assert by_hour.loc[12, "renewable_generation_mw"] > by_hour.loc[0, "renewable_generation_mw"]
    assert "voltage_anomaly" in set(frame["anomaly_label"])
    assert "frequency_anomaly" in set(frame["anomaly_label"])
    assert "line_overload" in set(frame["anomaly_label"])
    assert "fault_event" in set(frame["anomaly_label"])


@pytest.mark.anyio
async def test_generate_endpoint_persists_csv_and_history_endpoints_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sensor_data.csv"
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(csv_path))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        generate_response = await client.post(
            "/api/v1/simulation/generate?hours=24&interval_minutes=60&seed=7"
        )
        latest_response = await client.get("/api/v1/sensors/latest")
        history_response = await client.get("/api/v1/sensors/history?hours=6")

    assert generate_response.status_code == 200
    assert csv_path.exists()
    generated = generate_response.json()
    assert generated["status"] == "ok"
    assert generated["record_count"] > 0

    assert latest_response.status_code == 200
    latest_payload = latest_response.json()
    assert latest_payload["status"] == "ok"
    assert set(latest_payload["record"]) == set(REQUIRED_COLUMNS)

    assert history_response.status_code == 200
    history_payload = history_response.json()
    assert history_payload["status"] == "ok"
    assert history_payload["hours"] == 6
    assert history_payload["count"] > 0
    assert len(history_payload["records"]) == history_payload["count"]
