"""Shared settings and runtime dataclasses for GridPulse."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AppSettings:
    name: str
    tick_interval_seconds: int
    random_seed: int


@dataclass(frozen=True)
class GridSettings:
    mode: str
    zone_count: int
    topology_path: str
    profile_path: str


@dataclass(frozen=True)
class ScenarioSettings:
    default: str
    renewable_surplus_threshold: float
    available: List[str]


@dataclass(frozen=True)
class MLSettings:
    forecast_model: str
    classifier: str
    anomaly_detector: str
    use_lstm: bool


@dataclass(frozen=True)
class ValidationSettings:
    enable_pandapower: bool


@dataclass(frozen=True)
class ReferenceSettings:
    baseline: str


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    grid: GridSettings
    scenario: ScenarioSettings
    ml: MLSettings
    validation: ValidationSettings
    references: ReferenceSettings


@dataclass
class Suggestion:
    priority: str
    title: str
    why: str
    est_impact: str
    confidence: float
    category: str
    top_features: List[tuple] = field(default_factory=list)


@dataclass
class ZoneState:
    zone_id: str
    current_load_kw: float
    forecast_load_kw: float
    capacity_kw: float
    solar_kw: float
    wind_kw: float
    battery_soc: float
    temperature_c: float
    anomaly_score: float = 0.0


@dataclass
class ScenarioEvent:
    name: str
    description: str


@dataclass
class GridStatePrediction:
    label: str
    probabilities: Dict[str, float]


@dataclass
class DispatchPlan:
    total_demand_kw: float
    total_supplied_kw: float
    renewable_share: float
    max_line_utilization: float
    transmission_loss_kw: float
    overload_penalty_kw: float
    renewable_bonus_kw: float
    objective_score: float
    degraded: bool
    line_flows: List[Dict[str, float]] = field(default_factory=list)
    overloaded_lines: List[Dict[str, float]] = field(default_factory=list)
    routes: Dict[str, List[str]] = field(default_factory=dict)
    curtailed_zone_ids: List[str] = field(default_factory=list)
    validation: Optional[Dict[str, object]] = None


@dataclass
class EngineSnapshot:
    tick: int
    timestamp: str
    scenario: ScenarioEvent
    zone_states: List[ZoneState]
    prediction: GridStatePrediction
    dispatch: DispatchPlan
    suggestions: List[Suggestion]
    anomaly_score: float
    faulted_nodes: List[str] = field(default_factory=list)
