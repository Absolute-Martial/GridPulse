"""Health route for the GridPulse backend."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "gridpulse-backend",
        "mode": "offline",
        "data_source": "synthetic-csv",
    }
