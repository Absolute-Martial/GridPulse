"""Optimization API routes backed by pandapower validation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from app.optimization.opf_solver import PandapowerOPFSolver
from app.synthetic.csv_adapter import CsvSchemaError, load_sensor_frame

router = APIRouter(tags=["optimization"])
solver = PandapowerOPFSolver()


class DispatchValidationRequest(BaseModel):
    dispatch_plan: dict[str, Any] = Field(default_factory=dict)


@router.post("/optimization/power-flow")
async def run_power_flow() -> dict[str, Any]:
    frame = _load_sensor_frame_or_raise()
    try:
        report = solver.run_power_flow(sensor_dataframe=frame)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "report": report}


@router.post("/optimization/opf")
async def run_opf() -> dict[str, Any]:
    frame = _load_sensor_frame_or_raise()
    try:
        report = solver.run_opf(sensor_dataframe=frame)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "report": report}


@router.post("/optimization/validate-dispatch")
async def validate_dispatch(
    payload: DispatchValidationRequest = Body(default_factory=DispatchValidationRequest),
) -> dict[str, Any]:
    frame = _load_sensor_frame_or_raise()
    try:
        report = solver.validate_dispatch(sensor_dataframe=frame, dispatch_plan=payload.dispatch_plan)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "report": report}


@router.get("/optimization/feasibility-report")
async def get_feasibility_report() -> dict[str, Any]:
    frame = _load_sensor_frame_or_raise()
    try:
        report = solver.last_report or solver.run_power_flow(sensor_dataframe=frame)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "report": report}


def _load_sensor_frame_or_raise():
    try:
        frame = load_sensor_frame()
    except CsvSchemaError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if frame is None:
        raise HTTPException(status_code=400, detail="No synthetic sensor history is available.")
    return frame
