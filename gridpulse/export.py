"""Export GridPulse snapshots for external UI consumers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gridpulse.engine import GridPulseEngine


def build_snapshot_payload(engine: GridPulseEngine, scenarios: list[str] | None = None) -> dict:
    scenarios = scenarios or ["normal", "peak", "renewable_surplus", "fault"]
    snapshots = []

    for scenario in scenarios:
        engine.set_scenario("normal")
        for faulted in list(engine.simulator.faulted_nodes):
            engine.clear_fault(faulted)
        if scenario == "fault":
            engine.inject_fault("S2")
        else:
            engine.set_scenario(scenario)
        snapshot = engine.advance_tick()
        snapshots.append(
            {
                "scenario": scenario,
                "timestamp": snapshot.timestamp,
                "state_label": snapshot.prediction.label,
                "anomaly_score": round(snapshot.anomaly_score, 3),
                "demand_kw": round(snapshot.dispatch.total_demand_kw, 2),
                "supplied_kw": round(snapshot.dispatch.total_supplied_kw, 2),
                "renewable_share": round(snapshot.dispatch.renewable_share, 3),
                "max_line_utilization": round(snapshot.dispatch.max_line_utilization, 2),
                "objective_score": round(snapshot.dispatch.objective_score, 2),
                "transmission_loss_kw": round(snapshot.dispatch.transmission_loss_kw, 2),
                "overload_penalty_kw": round(snapshot.dispatch.overload_penalty_kw, 2),
                "faulted_nodes": snapshot.faulted_nodes,
                "top_suggestions": [
                    {
                        "priority": suggestion.priority,
                        "title": suggestion.title,
                        "why": suggestion.why,
                        "impact": suggestion.est_impact,
                        "confidence": round(suggestion.confidence, 3),
                    }
                    for suggestion in snapshot.suggestions
                ],
                "zones": [
                    {
                        "zone_id": zone.zone_id,
                        "current_load_kw": round(zone.current_load_kw, 2),
                        "forecast_load_kw": round(zone.forecast_load_kw, 2),
                        "capacity_kw": round(zone.capacity_kw, 2),
                        "battery_soc": round(zone.battery_soc, 3),
                    }
                    for zone in snapshot.zone_states
                ],
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "gridpulse-project",
        "title": "GridPulse Snapshot Feed",
        "baseline": "OpenEMS-adapted UI",
        "scenarios": snapshots,
    }


def write_snapshot_file(engine: GridPulseEngine, target: str | Path, scenarios: list[str] | None = None) -> Path:
    payload = build_snapshot_payload(engine, scenarios=scenarios)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
