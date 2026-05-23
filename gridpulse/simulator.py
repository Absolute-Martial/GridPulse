"""Scenario-driven simulator for the hackathon GridPulse demo."""
from __future__ import annotations

from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd

from gridpulse.forecasting import ForecastModel
from gridpulse.schemas import ScenarioEvent, ZoneState
from gridpulse.topology import zone_capacity_map, zone_ids

SCENARIO_DESCRIPTIONS = {
    "normal": "Balanced daytime operation with moderate demand.",
    "peak": "Peak demand surge with elevated operator stress.",
    "renewable_surplus": "Solar-heavy interval with excess renewable generation.",
    "fault": "Fault response mode with degraded routing flexibility.",
}


class Simulator:
    def __init__(
        self,
        graph: nx.Graph,
        profiles: dict,
        forecaster: ForecastModel,
        seed: int = 7,
        start_time: datetime | None = None,
        default_scenario: str = "normal",
    ):
        self.graph = graph
        self.profiles = profiles
        self.forecaster = forecaster
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.start_time = start_time or datetime(2026, 5, 22, 9, 0, 0)
        self.scenario_name = default_scenario
        self.faulted_nodes: list[str] = []
        self.zone_caps = zone_capacity_map(graph)

    def set_scenario(self, name: str) -> None:
        self.scenario_name = name

    def inject_fault(self, node_id: str) -> None:
        if node_id not in self.faulted_nodes:
            self.faulted_nodes.append(node_id)
            self.scenario_name = "fault"

    def clear_fault(self, node_id: str) -> None:
        if node_id in self.faulted_nodes:
            self.faulted_nodes.remove(node_id)
        if not self.faulted_nodes and self.scenario_name == "fault":
            self.scenario_name = "normal"

    def advance(self) -> tuple[str, ScenarioEvent, list[ZoneState], pd.Series]:
        hour = self.tick % 24
        next_hour = (hour + 1) % 24
        temperature = float(self.profiles["temperature_c"][hour])
        solar_factor = float(self.profiles["solar_factor"][hour])
        wind_factor = float(self.profiles["wind_factor"][hour])
        battery_soc = float(np.clip(0.52 + 0.25 * solar_factor - 0.12 * (self.scenario_name == "peak"), 0.18, 0.95))
        scenario_factor = self._scenario_factor()
        scenario_code = self._scenario_code()

        zones: list[ZoneState] = []
        total_demand = 0.0
        for zone_id in zone_ids(self.graph):
            attrs = self.graph.nodes[zone_id]
            base = float(attrs.get("base_load_kw", 120.0))
            multiplier = float(self.profiles["zones"][zone_id][hour])
            current = base * multiplier * scenario_factor
            current *= 1.0 + float(self.rng.normal(0.0, 0.015))
            current += max(0.0, temperature - 24.0) * 2.8
            current = max(10.0, round(current, 2))
            total_demand += current
            forecast = self.forecaster.predict_zone_load(
                zone_id=zone_id,
                hour=next_hour,
                current_load_kw=current,
                temperature_c=float(self.profiles["temperature_c"][next_hour]),
                solar_factor=float(self.profiles["solar_factor"][next_hour]),
                wind_factor=float(self.profiles["wind_factor"][next_hour]),
                scenario_code=scenario_code,
            )
            zones.append(
                ZoneState(
                    zone_id=zone_id,
                    current_load_kw=current,
                    forecast_load_kw=max(10.0, round(forecast, 2)),
                    capacity_kw=self.zone_caps[zone_id],
                    solar_kw=round(base * solar_factor * 0.22, 2),
                    wind_kw=round(base * wind_factor * 0.14, 2),
                    battery_soc=battery_soc,
                    temperature_c=temperature,
                )
            )

        feature_row = self._uci_like_features(zones, total_demand, solar_factor, wind_factor)
        event = ScenarioEvent(
            name=self.scenario_name,
            description=SCENARIO_DESCRIPTIONS[self.scenario_name],
        )
        timestamp = (self.start_time + timedelta(minutes=15 * self.tick)).isoformat()
        self.tick += 1
        return timestamp, event, zones, feature_row

    def _scenario_factor(self) -> float:
        return {
            "normal": 1.0,
            "peak": 1.22,
            "renewable_surplus": 0.93,
            "fault": 1.08,
        }[self.scenario_name]

    def _scenario_code(self) -> int:
        return {
            "normal": 0,
            "peak": 1,
            "renewable_surplus": 2,
            "fault": 1,
        }[self.scenario_name]

    def _uci_like_features(self, zones: list[ZoneState], total_demand: float, solar_factor: float, wind_factor: float) -> pd.Series:
        avg_load_ratio = np.mean([zone.current_load_kw / zone.capacity_kw for zone in zones])
        forecast_ratio = np.mean([zone.forecast_load_kw / zone.capacity_kw for zone in zones])
        renewable_factor = solar_factor + wind_factor
        tau_base = max(0.55, 2.6 - renewable_factor)
        p_base = np.clip((avg_load_ratio - 0.78) * 3.2, -2.0, 2.0)
        g_base = np.clip(0.18 + renewable_factor * 0.38 - avg_load_ratio * 0.12, 0.05, 1.0)

        values = {}
        for idx in range(1, 5):
            values[f"tau{idx}"] = round(tau_base + (idx - 2.5) * 0.08 + 0.05 * (self.scenario_name == "fault"), 4)
            values[f"p{idx}"] = round(np.clip(p_base + (idx - 2.5) * 0.11 + 0.18 * (self.scenario_name == "peak"), -2.0, 2.0), 4)
            values[f"g{idx}"] = round(np.clip(g_base + (2.5 - idx) * 0.04 + 0.06 * (self.scenario_name == "renewable_surplus"), 0.05, 1.0), 4)

        stab = round(0.03 - avg_load_ratio * 0.08 + renewable_factor * 0.03 - 0.04 * (self.scenario_name == "fault"), 4)
        stabf = "unstable" if forecast_ratio > 0.92 or self.scenario_name == "fault" else "stable"
        values["stab"] = stab
        values["stabf"] = stabf
        return pd.Series(values)
