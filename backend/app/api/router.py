"""Top-level API router."""

from fastapi import APIRouter

from app.api.routes.anomaly import router as anomaly_router
from app.api.explainability import router as explainability_router
from app.api.routes.forecast import router as forecast_router
from app.api.routes.gnn import router as gnn_router
from app.api.routes.grid import router as grid_router
from app.api.routes.health import router as health_router
from app.api.routes.optimization import router as optimization_router
from app.api.routes.recommendations import router as recommendations_router
from app.api.routes.sensors import router as sensors_router
from app.api.routes.simulation import router as simulation_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(explainability_router, prefix="/api/v1")
api_router.include_router(anomaly_router, prefix="/api/v1")
api_router.include_router(forecast_router, prefix="/api/v1")
api_router.include_router(gnn_router, prefix="/api/v1")
api_router.include_router(grid_router, prefix="/api/v1")
api_router.include_router(optimization_router, prefix="/api/v1")
api_router.include_router(recommendations_router, prefix="/api/v1")
api_router.include_router(sensors_router, prefix="/api/v1")
api_router.include_router(simulation_router, prefix="/api/v1")
