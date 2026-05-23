"""Configuration loader for the hackathon-first GridPulse build."""
from __future__ import annotations

from pathlib import Path

import yaml

from gridpulse.schemas import (
    AppSettings,
    GridSettings,
    MLSettings,
    ReferenceSettings,
    ScenarioSettings,
    Settings,
    ValidationSettings,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "defaults.yaml"


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(config_path.read_text())
    return Settings(
        app=AppSettings(**raw["app"]),
        grid=GridSettings(**raw["grid"]),
        scenario=ScenarioSettings(**raw["scenario"]),
        ml=MLSettings(**raw["ml"]),
        validation=ValidationSettings(**raw["validation"]),
        references=ReferenceSettings(**raw["references"]),
    )
