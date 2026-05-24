"""Explainability API routes for forecast models."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.forecasting import _load_ami_history_or_raise
from app.explainability.shap_forecast_explainer import explain_tree_forecast
from app.forecasting.schema import ForecastEntityError, ForecastUntrainedModelError
from app.forecasting.tree_forecaster import TreeForecaster

router = APIRouter(tags=["explainability"])


@router.get("/explain/forecast")
async def explain_forecast(
    feeder_id: str | None = Query(default=None),
    substation_id: str | None = Query(default=None),
    enterprise_id: str | None = Query(default=None),
    model: str = Query(default="tree", min_length=1),
    horizon: str = Query(default="1h", min_length=2),
) -> dict:
    if model.strip().lower() != "tree":
        raise HTTPException(status_code=400, detail={"error": "invalid_model", "message": "Only tree explanations are supported in this phase."})

    if feeder_id:
        entity_type, entity_id = "feeder", feeder_id
    elif substation_id:
        entity_type, entity_id = "substation", substation_id
    elif enterprise_id:
        entity_type, entity_id = "enterprise", enterprise_id
    else:
        raise HTTPException(status_code=404, detail={"error": "missing_entity", "message": "A forecast target is required."})

    history = _load_ami_history_or_raise()
    forecaster = TreeForecaster()
    try:
        explanation = explain_tree_forecast(
            forecaster=forecaster,
            dataframe=history,
            entity_type=entity_type,
            entity_id=entity_id,
            horizon=horizon,
        )
    except (ForecastEntityError, ForecastUntrainedModelError) as exc:
        raise HTTPException(status_code=400, detail={"error": "explainability_error", "message": str(exc)}) from exc
    return {"status": "ok", "explanation": explanation}
