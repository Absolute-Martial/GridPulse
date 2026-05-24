"""Recommendation routes."""

from fastapi import APIRouter, HTTPException

from app.recommendations.recommendation_engine import GridRecommendationEngine
from app.synthetic.csv_adapter import CsvSchemaError, load_sensor_frame

router = APIRouter(tags=["recommendations"])


@router.get("/recommendations")
async def get_recommendations() -> dict:
    try:
        frame = load_sensor_frame()
    except CsvSchemaError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if frame is None:
        raise HTTPException(status_code=400, detail="No synthetic sensor history is available.")

    engine = GridRecommendationEngine()
    recommendations = engine.generate_recommendations(frame)
    return {
        "status": "ok",
        "count": len(recommendations),
        "recommendations": recommendations,
    }
