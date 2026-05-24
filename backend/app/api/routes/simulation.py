"""Simulation routes for generating synthetic telemetry."""

from fastapi import APIRouter, Query

from app.simulator.sensor_generator import generate_and_save_sensor_data

router = APIRouter(tags=["simulation"])


@router.post("/simulation/generate")
async def generate_simulation(
    hours: int = Query(default=24, ge=1, le=168),
    interval_minutes: int = Query(default=5, ge=1, le=60),
    seed: int = Query(default=42, ge=0),
) -> dict:
    frame, output_path = generate_and_save_sensor_data(
        hours=hours,
        interval_minutes=interval_minutes,
        seed=seed,
    )
    timestamps = frame["timestamp"]
    return {
        "status": "ok",
        "source": "synthetic_generator",
        "path": str(output_path),
        "hours": hours,
        "interval_minutes": interval_minutes,
        "record_count": len(frame),
        "start_timestamp": timestamps.iloc[0],
        "end_timestamp": timestamps.iloc[-1],
    }
