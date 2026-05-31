from __future__ import annotations

import math

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from app.forecasting.schema import CANONICAL_INTERVAL_HOURS, ensure_canonical_history
from app.main import app
from app.simulator.ami_history_generator import generate_ami_history
from app.simulator.continuous_ami_generator import append_continuous_ami_history
from app.forecasting.continuous_training import run_continuous_training_cycle


def test_continuous_generator_appends_future_15_minute_rows() -> None:
    base = generate_ami_history(days=2, seed=11)
    updated = append_continuous_ami_history(base, steps=4, seed=12)

    assert len(updated) == len(base) + 4 * base["entity_id"].nunique()
    ensure_canonical_history(updated)
    assert pd.to_datetime(updated["timestamp"], utc=True).max() > pd.to_datetime(base["timestamp"], utc=True).max()


def test_continuous_generator_keeps_grid_physics_bounds() -> None:
    base = generate_ami_history(days=2, seed=11)
    updated = append_continuous_ami_history(base, steps=2, seed=12)
    generated = updated.tail(2 * base["entity_id"].nunique()).copy()

    for _, row in generated.iterrows():
        assert math.isclose(row["interval_energy_kwh"], row["load_kw"] * CANONICAL_INTERVAL_HOURS, rel_tol=0.0, abs_tol=0.002)
        assert row["load_kw"] <= row["contracted_md_kw"] * 1.18
        assert row["load_kw"] >= 5.0

    substation_rows = generated[generated["entity_type"] == "substation"]
    feeder_rows = generated[generated["entity_type"] == "feeder"]
    assert not substation_rows.empty
    assert not feeder_rows.empty
    assert float(substation_rows["load_kw"].sum()) >= float(feeder_rows["load_kw"].sum()) * 0.8


def test_continuous_training_cycle_writes_artifact(tmp_path) -> None:
    history_path = tmp_path / "ami_history.csv"
    fingerprint_path = tmp_path / "fingerprints.csv"
    artifact_dir = tmp_path / "models"
    base = generate_ami_history(days=5, seed=21)
    base.to_csv(history_path, index=False)

    result = run_continuous_training_cycle(
        history_path=history_path,
        fingerprint_path=fingerprint_path,
        artifact_dir=artifact_dir,
        steps=4,
        seed=22,
        model="tree",
        horizon="1h",
        entity_type="feeder",
        entity_id="FD_RES_01",
    )

    assert result["status"] == "ok"
    assert result["generated_rows"] > 0
    assert result["history_rows"] > len(base)
    assert fingerprint_path.exists()
    assert (artifact_dir / "tree_feeder_FD_RES_01_1h.pkl").exists()


@pytest.mark.anyio
async def test_continuous_api_endpoints_return_status(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "ami_history.csv"
    fingerprint_path = tmp_path / "fingerprints.csv"
    artifact_dir = tmp_path / "models"
    monkeypatch.setenv("GRIDPULSE_AMI_HISTORY_PATH", str(history_path))
    monkeypatch.setenv("GRIDPULSE_FORECAST_FINGERPRINT_PATH", str(fingerprint_path))
    monkeypatch.setenv("GRIDPULSE_FORECAST_ARTIFACT_DIR", str(artifact_dir))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        generate_response = await client.post("/api/v1/forecast/continuous/generate?steps=4&seed=42")
        train_response = await client.post(
            "/api/v1/forecast/continuous/train-cycle?steps=4&seed=43&model=tree&horizon=1h&entity_type=feeder&entity_id=FD_RES_01"
        )
        status_response = await client.get("/api/v1/forecast/continuous/status")

    assert generate_response.status_code == 200
    assert generate_response.json()["status"] == "ok"

    assert train_response.status_code == 200
    assert train_response.json()["status"] == "ok"

    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] == "ok"
    assert payload["history_exists"] is True
    assert payload["fingerprint_exists"] is True
