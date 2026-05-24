"""Grid digital twin routes."""

from fastapi import APIRouter, HTTPException, Query

from app.digital_twin.grid_graph import (
    clear_fault,
    find_alternative_path,
    get_grid_summary,
    get_overloaded_lines,
    inject_fault,
    serialize_grid_state,
    update_grid_state,
)
from app.synthetic.csv_adapter import CsvSchemaError, load_sensor_frame

router = APIRouter(tags=["grid"])


@router.get("/grid/state")
async def grid_state() -> dict:
    _refresh_grid_state()
    return {
        "status": "ok",
        **serialize_grid_state(),
    }


@router.get("/grid/summary")
async def grid_summary() -> dict:
    _refresh_grid_state()
    return {
        "status": "ok",
        "summary": get_grid_summary(),
    }


@router.get("/grid/overloads")
async def grid_overloads() -> dict:
    _refresh_grid_state()
    return {
        "status": "ok",
        "overloads": get_overloaded_lines(),
    }


@router.post("/grid/fault/inject")
async def grid_fault_inject(line_id: str = Query(..., min_length=1)) -> dict:
    try:
        edge = inject_fault(line_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "ok",
        "edge": edge,
    }


@router.post("/grid/fault/clear")
async def grid_fault_clear(line_id: str = Query(..., min_length=1)) -> dict:
    try:
        edge = clear_fault(line_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "ok",
        "edge": edge,
    }


@router.get("/grid/alternative-path")
async def grid_alternative_path(
    source_bus: str = Query(..., min_length=1),
    target_bus: str = Query(..., min_length=1),
) -> dict:
    path = find_alternative_path(source_bus=source_bus, target_bus=target_bus)
    return {
        "status": "ok",
        "source_bus": source_bus,
        "target_bus": target_bus,
        "path": path,
    }


def _refresh_grid_state() -> None:
    try:
        frame = load_sensor_frame()
    except CsvSchemaError:
        raise

    if frame is not None:
        update_grid_state(frame)
