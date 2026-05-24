from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.simulator.sensor_generator import generate_sensor_data, save_sensor_data


def test_recommendation_engine_returns_ranked_rule_based_actions(tmp_path: Path) -> None:
    from app.recommendations.recommendation_engine import GridRecommendationEngine

    frame = generate_sensor_data(hours=120, interval_minutes=60, seed=7)
    forecast_model_path = tmp_path / "load_forecaster.pkl"
    anomaly_model_path = tmp_path / "anomaly_detector.pkl"

    engine = GridRecommendationEngine(
        forecast_model_path=forecast_model_path,
        anomaly_model_path=anomaly_model_path,
    )
    recommendations = engine.generate_recommendations(frame)

    assert recommendations
    required_keys = {
        "recommendation_id",
        "priority",
        "action",
        "reason",
        "affected_zone",
        "affected_line",
        "expected_impact",
        "confidence_score",
        "validation_status",
    }
    assert required_keys <= set(recommendations[0])
    priorities = [item["priority"] for item in recommendations]
    assert priorities == sorted(
        priorities,
        key=lambda value: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[value],
    )
    actions = {item["action"] for item in recommendations}
    assert "isolate_fault" in actions
    assert "reroute_power" in actions


@pytest.mark.anyio
async def test_recommendations_endpoint_returns_rule_based_ranked_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sensor_data.csv"
    forecast_model_path = tmp_path / "load_forecaster.pkl"
    anomaly_model_path = tmp_path / "anomaly_detector.pkl"
    frame = generate_sensor_data(hours=120, interval_minutes=60, seed=7)
    save_sensor_data(frame, csv_path=csv_path)
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(csv_path))
    monkeypatch.setenv("GRIDPULSE_FORECAST_MODEL_PATH", str(forecast_model_path))
    monkeypatch.setenv("GRIDPULSE_ANOMALY_MODEL_PATH", str(anomaly_model_path))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/v1/recommendations")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["count"] >= 2
    assert isinstance(payload["recommendations"], list)
    assert any(item["action"] == "isolate_fault" for item in payload["recommendations"])
    assert any(item["action"] == "reroute_power" for item in payload["recommendations"])
