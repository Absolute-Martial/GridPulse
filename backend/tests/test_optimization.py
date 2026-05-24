from pathlib import Path

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.digital_twin.grid_graph import build_default_grid, update_grid_state
from app.simulator.sensor_generator import generate_sensor_data, save_sensor_data


pytest.importorskip("pandapower")


def _normalized_frame() -> pd.DataFrame:
    frame = generate_sensor_data(hours=48, interval_minutes=60, seed=7)
    timestamp_series = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = timestamp_series.max()
    latest_mask = timestamp_series == latest_timestamp
    frame.loc[latest_mask, "fault_status"] = "normal"
    frame.loc[latest_mask, "line_flow_mw"] = frame.loc[latest_mask, "line_capacity_mw"] * 0.55
    frame.loc[latest_mask, "voltage_pu"] = 1.0
    frame.loc[latest_mask, "frequency_hz"] = 50.0
    return frame


def test_build_network_from_default_graph() -> None:
    from app.optimization.pandapower_adapter import build_pandapower_network

    frame = _normalized_frame()
    graph = build_default_grid()
    update_grid_state(frame)

    net, metadata = build_pandapower_network(graph, frame)

    assert len(net.bus) >= 8
    assert len(net.line) >= 10
    assert len(net.ext_grid) == 1
    assert metadata["bus_map"]
    assert metadata["line_map"]


def test_run_power_flow_on_normal_scenario() -> None:
    from app.optimization.opf_solver import PandapowerOPFSolver

    frame = _normalized_frame()
    solver = PandapowerOPFSolver()
    report = solver.run_power_flow(sensor_dataframe=frame)

    assert report["is_feasible"] is True
    assert "total_losses_mw" in report
    assert "voltage_violations" in report


def test_detect_overloaded_line() -> None:
    from app.optimization.opf_solver import PandapowerOPFSolver

    frame = _normalized_frame()
    timestamp_series = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = timestamp_series.max()
    mask = timestamp_series == latest_timestamp
    frame.loc[mask & (frame["line_id"] == "line-2"), "line_flow_mw"] = 60.0
    frame.loc[mask & (frame["line_id"] == "line-2"), "line_capacity_mw"] = 20.0

    solver = PandapowerOPFSolver()
    report = solver.run_power_flow(sensor_dataframe=frame)

    assert any(item["line_id"] == "line-2" for item in report["overloaded_lines"])


def test_detect_voltage_violation() -> None:
    from app.optimization.opf_solver import PandapowerOPFSolver

    frame = _normalized_frame()
    timestamp_series = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = timestamp_series.max()
    mask = timestamp_series == latest_timestamp
    frame.loc[mask & (frame["bus_id"] == "bus-3"), "voltage_pu"] = 1.12

    solver = PandapowerOPFSolver()
    report = solver.run_power_flow(sensor_dataframe=frame)

    assert any(item["bus_id"] == "bus-3" for item in report["voltage_violations"])


def test_graceful_failure_when_opf_cannot_converge() -> None:
    from app.optimization.opf_solver import PandapowerOPFSolver

    frame = _normalized_frame()
    solver = PandapowerOPFSolver()
    impossible_dispatch = {
        "ext_grid": {"max_p_mw": 0.1},
        "generators": {
            "gen-1": {"max_p_mw": 0.1},
            "gen-2": {"max_p_mw": 0.1},
        },
        "loads": {
            "zone-1": {"min_p_mw": 25.0},
            "zone-2": {"min_p_mw": 25.0},
            "zone-3": {"min_p_mw": 25.0},
            "zone-4": {"min_p_mw": 25.0},
            "zone-5": {"min_p_mw": 25.0},
        },
    }

    result = solver.validate_dispatch(sensor_dataframe=frame, dispatch_plan=impossible_dispatch)

    assert result["is_feasible"] is False
    assert result["failure_reason"]


@pytest.mark.anyio
async def test_optimization_api_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sensor_data.csv"
    frame = _normalized_frame()
    save_sensor_data(frame, csv_path=csv_path)
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(csv_path))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        pf_response = await client.post("/api/v1/optimization/power-flow")
        opf_response = await client.post("/api/v1/optimization/opf")
        validate_response = await client.post(
            "/api/v1/optimization/validate-dispatch",
            json={"dispatch_plan": {"ext_grid": {"max_p_mw": 200.0}}},
        )
        report_response = await client.get("/api/v1/optimization/feasibility-report")

    assert pf_response.status_code == 200
    assert opf_response.status_code == 200
    assert validate_response.status_code == 200
    assert report_response.status_code == 200
    assert pf_response.json()["status"] == "ok"
    assert "report" in report_response.json()
