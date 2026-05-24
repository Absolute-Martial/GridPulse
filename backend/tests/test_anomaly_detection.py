from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.simulator.sensor_generator import generate_sensor_data, save_sensor_data


def test_anomaly_training_and_detection_contract(tmp_path: Path) -> None:
    from app.anomaly.anomaly_detector import GridAnomalyDetector

    frame = generate_sensor_data(hours=120, interval_minutes=60, seed=7)
    model_path = tmp_path / "anomaly_detector.pkl"

    detector = GridAnomalyDetector(model_path=model_path)
    training_result = detector.train(frame)
    history = detector.detect_history(frame)
    latest = detector.detect_latest(frame)

    assert model_path.exists()
    assert training_result["artifact_path"] == str(model_path)
    assert training_result["train_rows"] > 0
    assert training_result["feature_count"] == 8

    assert history
    assert latest
    required_keys = {
        "timestamp",
        "sensor_id",
        "anomaly_score",
        "anomaly_type",
        "severity",
        "affected_zone",
        "explanation_text",
    }
    assert required_keys <= set(history[0])
    assert required_keys <= set(latest[0])
    assert any(item["anomaly_type"] == "line overload" for item in history)


@pytest.mark.anyio
async def test_anomaly_endpoints_train_latest_and_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sensor_data.csv"
    model_path = tmp_path / "anomaly_detector.pkl"
    frame = generate_sensor_data(hours=120, interval_minutes=60, seed=7)
    save_sensor_data(frame, csv_path=csv_path)
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(csv_path))
    monkeypatch.setenv("GRIDPULSE_ANOMALY_MODEL_PATH", str(model_path))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        train_response = await client.post("/api/v1/anomaly/train")
        latest_response = await client.get("/api/v1/anomaly/latest")
        history_response = await client.get("/api/v1/anomaly/history")

    assert train_response.status_code == 200
    assert latest_response.status_code == 200
    assert history_response.status_code == 200

    train_payload = train_response.json()
    latest_payload = latest_response.json()
    history_payload = history_response.json()

    assert train_payload["status"] == "ok"
    assert train_payload["artifact_path"] == str(model_path)
    assert model_path.exists()

    assert latest_payload["status"] == "ok"
    assert latest_payload["count"] >= 1
    assert isinstance(latest_payload["anomalies"], list)

    assert history_payload["status"] == "ok"
    assert history_payload["count"] >= latest_payload["count"]
    assert isinstance(history_payload["anomalies"], list)
