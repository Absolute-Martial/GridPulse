from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.anyio
async def test_forecasting_api_supports_target_aware_horizons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ami_history_path = tmp_path / "ami_history.csv"
    fingerprint_path = tmp_path / "fingerprints.csv"
    artifact_dir = tmp_path / "forecasting-artifacts"

    monkeypatch.setenv("GRIDPULSE_AMI_HISTORY_PATH", str(ami_history_path))
    monkeypatch.setenv("GRIDPULSE_FORECAST_FINGERPRINT_PATH", str(fingerprint_path))
    monkeypatch.setenv("GRIDPULSE_FORECAST_ARTIFACT_DIR", str(artifact_dir))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        history_response = await client.post("/api/v1/forecast/history/generate?days=21&seed=7")
        fingerprint_response = await client.post("/api/v1/forecast/fingerprint/build")
        train_response = await client.post(
            "/api/v1/forecast/train?feeder_id=FD_RES_01&horizon=1h&model=tree"
        )
        run_response = await client.get(
            "/api/v1/forecast/feeder?feeder_id=FD_RES_01&horizon=1h&model=tree"
        )
        evaluate_response = await client.get(
            "/api/v1/forecast/evaluate?feeder_id=FD_RES_01&horizon=1h&model=tree"
        )
        explain_response = await client.get(
            "/api/v1/explain/forecast?feeder_id=FD_RES_01&horizon=1h&model=tree"
        )
        models_response = await client.get("/api/v1/forecast/models")

    assert history_response.status_code == 200
    assert fingerprint_response.status_code == 200
    assert train_response.status_code == 200
    assert run_response.status_code == 200
    assert evaluate_response.status_code == 200
    assert explain_response.status_code == 200
    assert models_response.status_code == 200

    run_payload = run_response.json()
    evaluate_payload = evaluate_response.json()
    explain_payload = explain_response.json()
    models_payload = models_response.json()

    assert run_payload["forecast"]["horizon"] == "1h"
    assert len(run_payload["forecast"]["slot_predictions"]) == 4
    assert "aggregated_summary" in run_payload["forecast"]
    assert {"mae", "rmse", "mape", "r2", "peak_time_error", "peak_load_error"} <= set(
        evaluate_payload["evaluation"]["metrics"]
    )
    assert explain_payload["explanation"]["top_features"]
    assert {"fingerprint_baseline", "tree"} <= set(models_payload["models"])
