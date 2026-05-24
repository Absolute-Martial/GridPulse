from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.anyio
async def test_health_returns_offline_backend_status() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "gridpulse-backend",
        "mode": "offline",
        "data_source": "synthetic-csv",
    }


@pytest.mark.anyio
async def test_latest_sensor_returns_no_data_when_csv_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "missing.csv"
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(csv_path))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/v1/sensors/latest")

    assert response.status_code == 200
    assert response.json() == {
        "status": "no_data",
        "source": "synthetic_csv",
        "record": None,
    }


@pytest.mark.anyio
async def test_latest_sensor_returns_last_csv_record(
    monkeypatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "latest_readings.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp,sensor_id,zone_id,bus_id,voltage_pu,current_a,frequency_hz,load_mw,generation_mw,renewable_generation_mw,line_id,line_flow_mw,line_capacity_mw,temperature_c,humidity_percent,storage_soc_percent,fault_status,anomaly_label",
                "2026-05-24T09:00:00+00:00,sensor-line-1-a,zone-1,bus-1,1.01,210.0,50.0,6.2,4.0,1.2,line-1,8.0,22.0,24.0,56.0,68.0,normal,normal",
                "2026-05-24T09:05:00+00:00,sensor-line-2-b,zone-2,bus-3,0.99,245.0,49.98,7.4,5.5,1.8,line-2,9.4,24.0,25.0,54.0,63.0,normal,normal",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(csv_path))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/v1/sensors/latest")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "source": "synthetic_csv",
        "record": {
            "timestamp": "2026-05-24T09:05:00+00:00",
            "sensor_id": "sensor-line-2-b",
            "zone_id": "zone-2",
            "bus_id": "bus-3",
            "voltage_pu": 0.99,
            "current_a": 245.0,
            "frequency_hz": 49.98,
            "load_mw": 7.4,
            "generation_mw": 5.5,
            "renewable_generation_mw": 1.8,
            "line_id": "line-2",
            "line_flow_mw": 9.4,
            "line_capacity_mw": 24.0,
            "temperature_c": 25.0,
            "humidity_percent": 54.0,
            "storage_soc_percent": 63.0,
            "fault_status": "normal",
            "anomaly_label": "normal",
        },
    }
