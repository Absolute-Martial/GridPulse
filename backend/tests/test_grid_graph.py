from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.digital_twin.grid_graph import (
    build_default_grid,
    clear_fault,
    find_alternative_path,
    get_grid_summary,
    get_overloaded_lines,
    inject_fault,
    update_grid_state,
)
from app.simulator.sensor_generator import generate_sensor_data, save_sensor_data


def test_build_default_grid_creates_expected_nodes_and_edges() -> None:
    graph = build_default_grid()
    summary = get_grid_summary()

    assert summary["bus_count"] == 8
    assert summary["zone_count"] == 5
    assert summary["generator_count"] == 2
    assert summary["renewable_count"] == 2
    assert summary["battery_count"] == 1
    assert summary["load_count"] == 5
    assert summary["transmission_line_count"] == 10
    assert summary["transformer_count"] == 2
    assert graph.number_of_edges() == 12

    line_one = next(
        data for _, _, data in graph.edges(data=True) if data["line_id"] == "line-1"
    )
    assert {
        "line_id",
        "from_bus",
        "to_bus",
        "capacity_mw",
        "current_flow_mw",
        "impedance",
        "status",
        "loading_percent",
    } <= set(line_one)


def test_update_grid_state_sets_summary_and_overloads() -> None:
    build_default_grid()
    frame = generate_sensor_data(hours=24, interval_minutes=60, seed=7)

    graph = update_grid_state(frame)
    summary = get_grid_summary()
    overloads = get_overloaded_lines()

    assert graph.number_of_nodes() > 0
    assert summary["total_load_mw"] > 0
    assert summary["total_generation_mw"] > 0
    assert summary["failed_line_count"] >= 1
    assert any(overload["line_id"] == "line-8" for overload in overloads)


def test_fault_injection_and_alternative_path_search() -> None:
    graph = build_default_grid()

    inject_result = inject_fault("line-2")
    path = find_alternative_path("bus-2", "bus-4")
    clear_result = clear_fault("line-2")

    faulted_line = next(
        data for _, _, data in graph.edges(data=True) if data["line_id"] == "line-2"
    )

    assert inject_result["status"] == "failed"
    assert path == ["bus-2", "bus-5", "bus-4"]
    assert faulted_line["status"] == "active"
    assert clear_result["status"] == "active"


@pytest.mark.anyio
async def test_grid_endpoints_expose_state_summary_faults_and_overloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sensor_data.csv"
    frame = generate_sensor_data(hours=24, interval_minutes=60, seed=7)
    save_sensor_data(frame, csv_path=csv_path)
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(csv_path))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        state_response = await client.get("/api/v1/grid/state")
        summary_response = await client.get("/api/v1/grid/summary")
        overloads_response = await client.get("/api/v1/grid/overloads")
        inject_response = await client.post("/api/v1/grid/fault/inject?line_id=line-2")
        alternative_response = await client.get(
            "/api/v1/grid/alternative-path?source_bus=bus-2&target_bus=bus-4"
        )
        clear_response = await client.post("/api/v1/grid/fault/clear?line_id=line-2")

    assert state_response.status_code == 200
    assert summary_response.status_code == 200
    assert overloads_response.status_code == 200
    assert inject_response.status_code == 200
    assert alternative_response.status_code == 200
    assert clear_response.status_code == 200

    state_payload = state_response.json()
    summary_payload = summary_response.json()
    overloads_payload = overloads_response.json()

    assert state_payload["status"] == "ok"
    assert summary_payload["summary"]["bus_count"] == 8
    assert overloads_payload["status"] == "ok"
    assert any(item["line_id"] == "line-8" for item in overloads_payload["overloads"])
    assert inject_response.json()["edge"]["status"] == "failed"
    assert alternative_response.json()["path"] == ["bus-2", "bus-5", "bus-4"]
    assert clear_response.json()["edge"]["status"] == "active"
