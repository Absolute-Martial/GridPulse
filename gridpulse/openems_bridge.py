"""Backend snapshot export helpers."""
from __future__ import annotations

from pathlib import Path

from gridpulse.engine import GridPulseEngine
from gridpulse.export import write_snapshot_file


def default_snapshot_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "artifacts" / "gridpulse-snapshot.json"


def export_snapshot(
    engine: GridPulseEngine,
    scenarios: list[str] | None = None,
    target: str | Path | None = None,
) -> Path:
    path = Path(target) if target else default_snapshot_path()
    return write_snapshot_file(engine, target=path, scenarios=scenarios)
