"""Sensor routes backed by the synthetic CSV adapter."""

from fastapi import APIRouter, HTTPException, Query

from app.synthetic.csv_adapter import (
    CsvSchemaError,
    get_latest_sensor_record,
    get_sensor_history,
)

router = APIRouter(tags=["sensors"])


@router.get("/sensors/latest")
async def latest_sensor() -> dict:
    try:
        record = get_latest_sensor_record()
    except CsvSchemaError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if record is None:
        return {
            "status": "no_data",
            "source": "synthetic_csv",
            "record": None,
        }

    return {
        "status": "ok",
        "source": "synthetic_csv",
        "record": record,
    }


@router.get("/sensors/history")
async def sensor_history(hours: int = Query(default=24, ge=1, le=168)) -> dict:
    try:
        records = get_sensor_history(hours=hours)
    except CsvSchemaError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not records:
        return {
            "status": "no_data",
            "source": "synthetic_csv",
            "hours": hours,
            "count": 0,
            "records": [],
        }

    return {
        "status": "ok",
        "source": "synthetic_csv",
        "hours": hours,
        "count": len(records),
        "records": records,
    }
