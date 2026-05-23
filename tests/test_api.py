import pytest
from fastapi import HTTPException

from api.main import clear_fault, export_snapshot_endpoint, health, inject_fault, set_scenario, snapshot_feed, summary, tick


def test_health_endpoint_function():
    payload = health()
    assert payload["status"] == "ok"


def test_tick_function_returns_algorithm_snapshot():
    payload = tick()
    assert "dispatch" in payload
    assert "objective_score" in payload["dispatch"]
    assert "suggestions" in payload


def test_snapshot_function_validates_scenarios():
    payload = snapshot_feed(["normal", "fault"])
    assert len(payload["scenarios"]) == 2

    with pytest.raises(HTTPException):
        snapshot_feed(["invalid"])


def test_scenario_and_fault_controls():
    set_scenario("normal")
    inject_fault("S2")
    state = summary()
    assert "S2" in state["faulted_nodes"]
    clear_fault("S2")


def test_snapshot_export_endpoint_function():
    payload = export_snapshot_endpoint(["normal"])
    assert payload["target"].endswith("gridpulse-snapshot.json")
