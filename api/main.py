"""FastAPI backend for GridPulse algorithm services."""
from __future__ import annotations

from dataclasses import asdict
from threading import Lock

from fastapi import FastAPI, HTTPException, Query

from gridpulse.engine import GridPulseEngine
from gridpulse.export import build_snapshot_payload
from gridpulse.openems_bridge import export_snapshot

app = FastAPI(title="GridPulse Backend", version="1.0.0")
engine = GridPulseEngine()
engine_lock = Lock()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "gridpulse-backend"}


@app.get("/summary")
def summary() -> dict:
    return {
        "topology": engine.grid_summary(),
        "scenario": engine.simulator.scenario_name,
        "faulted_nodes": list(engine.simulator.faulted_nodes),
    }


@app.post("/scenario/{name}")
def set_scenario(name: str) -> dict:
    if name not in engine.settings.scenario.available:
        raise HTTPException(status_code=400, detail=f"Unsupported scenario: {name}")
    with engine_lock:
        engine.set_scenario(name)
    return {"scenario": name}


@app.post("/faults/{node_id}")
def inject_fault(node_id: str) -> dict:
    with engine_lock:
        engine.inject_fault(node_id)
    return {"faulted_nodes": list(engine.simulator.faulted_nodes)}


@app.delete("/faults/{node_id}")
def clear_fault(node_id: str) -> dict:
    with engine_lock:
        engine.clear_fault(node_id)
    return {"faulted_nodes": list(engine.simulator.faulted_nodes)}


@app.post("/tick")
def tick() -> dict:
    with engine_lock:
        snapshot = engine.advance_tick()
    return asdict(snapshot)


@app.get("/snapshot")
def snapshot_feed(scenarios: list[str] = Query(default=["normal", "peak", "renewable_surplus", "fault"])) -> dict:
    invalid = [item for item in scenarios if item not in engine.settings.scenario.available]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported scenarios: {', '.join(invalid)}")
    with engine_lock:
        payload = build_snapshot_payload(engine, scenarios=scenarios)
    return payload


@app.post("/integrations/snapshot/export")
def export_snapshot_endpoint(scenarios: list[str] = Query(default=["normal", "peak", "renewable_surplus", "fault"])) -> dict:
    invalid = [item for item in scenarios if item not in engine.settings.scenario.available]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported scenarios: {', '.join(invalid)}")
    with engine_lock:
        path = export_snapshot(engine, scenarios=scenarios)
    return {"target": str(path), "scenarios": scenarios}
