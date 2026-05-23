from gridpulse.engine import GridPulseEngine
from gridpulse.openems_bridge import default_snapshot_path, export_snapshot


def test_default_snapshot_path_points_to_repo_artifacts():
    path = default_snapshot_path()
    assert str(path).endswith("artifacts/gridpulse-snapshot.json")


def test_export_snapshot_to_openems_writes_json():
    engine = GridPulseEngine()
    path = export_snapshot(engine, scenarios=["normal"])
    assert path.exists()
    assert path.read_text().startswith("{")
