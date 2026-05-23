from pathlib import Path

from gridpulse.engine import GridPulseEngine
from gridpulse.export import build_snapshot_payload, write_snapshot_file


def test_build_snapshot_payload_contains_expected_scenarios():
    engine = GridPulseEngine()

    payload = build_snapshot_payload(engine, ["normal", "peak", "renewable_surplus", "fault"])

    assert payload["source"] == "gridpulse-project"
    assert len(payload["scenarios"]) == 4
    assert {item["scenario"] for item in payload["scenarios"]} == {
        "normal",
        "peak",
        "renewable_surplus",
        "fault",
    }
    fault_snapshot = next(item for item in payload["scenarios"] if item["scenario"] == "fault")
    assert fault_snapshot["faulted_nodes"]
    assert fault_snapshot["top_suggestions"]


def test_write_snapshot_file_creates_json(tmp_path: Path):
    engine = GridPulseEngine()
    target = tmp_path / "gridpulse-snapshot.json"

    write_snapshot_file(engine, target)

    assert target.exists()
    assert target.read_text().startswith("{")
