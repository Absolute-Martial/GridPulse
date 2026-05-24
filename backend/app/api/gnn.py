"""Topology-risk GNN API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.gnn.inference import GridRiskInference
from app.gnn.trainer import GridRiskGNNTrainer
from app.synthetic.csv_adapter import CsvSchemaError, load_sensor_frame

router = APIRouter(tags=["gnn"])


@router.post("/gnn/train")
async def train_gnn(
    epochs: int = Query(default=4, ge=1, le=50),
    scenarios_per_type: int = Query(default=2, ge=1, le=10),
) -> dict:
    frame = _load_frame_or_raise()
    trainer = GridRiskGNNTrainer()
    result = trainer.train(frame, epochs=epochs, scenarios_per_type=scenarios_per_type)
    return {"status": "ok", **result}


@router.get("/gnn/predict-risk")
async def predict_risk() -> dict:
    frame = _load_frame_or_raise()
    inference = GridRiskInference()
    try:
        risks = inference.predict_risk(frame)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", "count": len(risks), "risks": risks}


@router.get("/gnn/top-risk-lines")
async def top_risk_lines(limit: int = Query(default=5, ge=1, le=20)) -> dict:
    frame = _load_frame_or_raise()
    inference = GridRiskInference()
    try:
        lines = inference.top_risk_lines(frame, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", "count": len(lines), "top_risk_lines": lines}


@router.get("/gnn/top-risk-zones")
async def top_risk_zones(limit: int = Query(default=5, ge=1, le=20)) -> dict:
    frame = _load_frame_or_raise()
    inference = GridRiskInference()
    try:
        zones = inference.top_risk_zones(frame, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", "count": len(zones), "top_risk_zones": zones}


def _load_frame_or_raise():
    try:
        frame = load_sensor_frame()
    except CsvSchemaError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if frame is None:
        raise HTTPException(status_code=400, detail="No synthetic sensor history is available.")
    return frame
