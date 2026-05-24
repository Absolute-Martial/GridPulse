"""Forecasting API routes for the 15-minute AMI forecasting MVP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.forecasting.feeder_fingerprint import build_fingerprint_database
from app.forecasting.fingerprint_baseline import FingerprintBaselineForecaster
from app.forecasting.response_formatter import build_forecast_response
from app.forecasting.schema import (
    ForecastEntityError,
    ForecastFingerprintError,
    ForecastHistoryError,
    ForecastHorizonError,
    ForecastInsufficientHistoryError,
    ForecastUntrainedModelError,
    normalize_horizon,
)
from app.forecasting.tree_forecaster import TreeForecaster
from app.simulator.ami_history_generator import generate_ami_history, load_ami_history, save_ami_history

router = APIRouter(tags=["forecast"])

SUPPORTED_MODELS = ("fingerprint_baseline", "tree")


@router.post("/forecast/history/generate")
async def generate_forecast_history(
    days: int = Query(default=30, ge=1),
    seed: int = Query(default=7),
) -> dict:
    history = generate_ami_history(days=days, seed=seed)
    output_path = save_ami_history(history)
    return {
        "status": "ok",
        "rows": int(len(history)),
        "path": str(output_path),
    }


@router.post("/forecast/fingerprint/build")
async def build_forecast_fingerprint() -> dict:
    history = _load_ami_history_or_raise()
    fingerprint = build_fingerprint_database(history)
    output_path = _save_fingerprint_database(fingerprint)
    return {
        "status": "ok",
        "rows": int(len(fingerprint)),
        "path": str(output_path),
    }


@router.get("/forecast/fingerprint")
async def get_forecast_fingerprint(
    entity_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
) -> dict:
    fingerprint = _load_fingerprint_database_or_raise()
    if entity_type:
        fingerprint = fingerprint[fingerprint["entity_type"].astype(str).str.lower() == entity_type.strip().lower()]
    if entity_id:
        fingerprint = fingerprint[fingerprint["entity_id"].astype(str) == entity_id.strip()]
    return {
        "status": "ok",
        "records": fingerprint.sort_values(["entity_type", "entity_id", "day_of_week", "slot_index"]).to_dict(orient="records"),
    }


@router.post("/forecast/train")
async def train_forecast_model(
    model: str = Query(default="tree", min_length=1),
    horizon: str = Query(default="1h", min_length=2),
    entity_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    substation_id: str | None = Query(default=None),
    feeder_id: str | None = Query(default=None),
    enterprise_id: str | None = Query(default=None),
) -> dict:
    history = _load_ami_history_or_raise()
    target_type, target_id = _resolve_target_identity(
        entity_type=entity_type,
        entity_id=entity_id,
        substation_id=substation_id,
        feeder_id=feeder_id,
        enterprise_id=enterprise_id,
    )
    _normalize_horizon_or_raise(horizon)
    forecaster = _get_forecaster(model)
    try:
        result = forecaster.train(history, entity_type=target_type, entity_id=target_id, horizon=horizon)
    except (ForecastEntityError, ForecastHistoryError, ForecastFingerprintError, ForecastUntrainedModelError) as exc:
        _raise_structured_http_error(exc)
    return {
        "status": "ok",
        "model": model,
        "entity_type": target_type,
        "entity_id": target_id,
        **result,
    }


@router.get("/forecast/substation")
async def forecast_substation(
    substation_id: str = Query(..., min_length=1),
    horizon: str = Query(default="1h", min_length=2),
    model: str = Query(default="tree", min_length=1),
) -> dict:
    return _run_forecast_response("substation", substation_id, horizon, model)


@router.get("/forecast/feeder")
async def forecast_feeder(
    feeder_id: str = Query(..., min_length=1),
    horizon: str = Query(default="1h", min_length=2),
    model: str = Query(default="tree", min_length=1),
) -> dict:
    return _run_forecast_response("feeder", feeder_id, horizon, model)


@router.get("/forecast/enterprise")
async def forecast_enterprise(
    enterprise_id: str = Query(..., min_length=1),
    horizon: str = Query(default="1h", min_length=2),
    model: str = Query(default="tree", min_length=1),
) -> dict:
    return _run_forecast_response("enterprise", enterprise_id, horizon, model)


@router.get("/forecast/evaluate")
async def evaluate_forecast(
    model: str = Query(default="tree", min_length=1),
    horizon: str = Query(default="1h", min_length=2),
    entity_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    substation_id: str | None = Query(default=None),
    feeder_id: str | None = Query(default=None),
    enterprise_id: str | None = Query(default=None),
) -> dict:
    history = _load_ami_history_or_raise()
    target_type, target_id = _resolve_target_identity(
        entity_type=entity_type,
        entity_id=entity_id,
        substation_id=substation_id,
        feeder_id=feeder_id,
        enterprise_id=enterprise_id,
    )
    _normalize_horizon_or_raise(horizon)
    forecaster = _get_forecaster(model)
    try:
        evaluation = forecaster.evaluate(history, entity_type=target_type, entity_id=target_id, horizon=horizon)
    except (ForecastEntityError, ForecastHistoryError, ForecastFingerprintError, ForecastUntrainedModelError) as exc:
        _raise_structured_http_error(exc)
    return {
        "status": "ok",
        "model": model,
        "entity_type": target_type,
        "entity_id": target_id,
        "evaluation": evaluation,
    }


@router.get("/forecast/models")
async def forecast_models() -> dict:
    return {"status": "ok", "models": list(SUPPORTED_MODELS)}


def _run_forecast_response(entity_type: str, entity_id: str, horizon: str, model: str) -> dict:
    history = _load_ami_history_or_raise()
    normalized_horizon, _ = _normalize_horizon_or_raise(horizon)
    effective_model = model
    filtered_history = history[
        (history["entity_type"].astype(str).str.lower() == entity_type)
        & (history["entity_id"].astype(str) == entity_id)
    ].copy()
    if filtered_history.empty:
        _raise_structured_http_error(ForecastEntityError(f"Unknown {entity_type} target: {entity_id}."))
    if len(filtered_history) < 96 and model != "fingerprint_baseline":
        effective_model = "fingerprint_baseline"

    forecaster = _get_forecaster(effective_model)
    try:
        raw_forecast = forecaster.predict(history, entity_type=entity_type, entity_id=entity_id, horizon=normalized_horizon)
    except (ForecastEntityError, ForecastHistoryError, ForecastFingerprintError, ForecastUntrainedModelError) as exc:
        _raise_structured_http_error(exc)

    confidence = "low" if len(filtered_history) < 480 else raw_forecast.get("summary", {}).get("confidence", "medium")
    forecast = build_forecast_response(
        entity_type=entity_type,
        entity_id=entity_id,
        horizon=normalized_horizon,
        latest_timestamp=raw_forecast["latest_input_timestamp"],
        slot_predictions=raw_forecast["slot_predictions"],
        confidence=confidence,
        model_name=effective_model,
    )
    return {
        "status": "ok",
        "forecast": forecast,
    }


def _get_forecaster(model: str):
    normalized = model.strip().lower()
    if normalized == "fingerprint_baseline":
        return FingerprintBaselineForecaster(model_dir=get_settings().forecast_artifact_dir)
    if normalized == "tree":
        return TreeForecaster(model_dir=get_settings().forecast_artifact_dir)
    raise HTTPException(status_code=400, detail={"error": "invalid_model", "message": f"Unsupported model '{model}'."})


def _load_ami_history_or_raise() -> pd.DataFrame:
    history = load_ami_history()
    if history is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_history", "message": "No AMI history is available. Generate history first."},
        )
    return history


def _save_fingerprint_database(fingerprint: pd.DataFrame) -> Path:
    path = get_settings().forecast_fingerprint_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint.to_csv(path, index=False)
    return path


def _load_fingerprint_database_or_raise() -> pd.DataFrame:
    path = get_settings().forecast_fingerprint_path
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_fingerprint", "message": "No fingerprint database is available. Build it first."},
        )
    return pd.read_csv(path)


def _resolve_target_identity(
    *,
    entity_type: str | None,
    entity_id: str | None,
    substation_id: str | None,
    feeder_id: str | None,
    enterprise_id: str | None,
) -> tuple[str, str]:
    if entity_type and entity_id:
        return entity_type.strip().lower(), entity_id.strip()
    if substation_id:
        return "substation", substation_id.strip()
    if feeder_id:
        return "feeder", feeder_id.strip()
    if enterprise_id:
        return "enterprise", enterprise_id.strip()
    _raise_structured_http_error(ForecastEntityError("A forecast target is required."))


def _normalize_horizon_or_raise(horizon: str) -> tuple[str, int]:
    try:
        return normalize_horizon(horizon)
    except ForecastHorizonError as exc:
        _raise_structured_http_error(exc)


def _raise_structured_http_error(exc: Exception) -> None:
    if isinstance(exc, ForecastHorizonError):
        raise HTTPException(status_code=400, detail={"error": "invalid_horizon", "message": str(exc)}) from exc
    if isinstance(exc, ForecastEntityError):
        raise HTTPException(status_code=404, detail={"error": "missing_entity", "message": str(exc)}) from exc
    if isinstance(exc, ForecastInsufficientHistoryError):
        raise HTTPException(status_code=400, detail={"error": "insufficient_history", "message": str(exc)}) from exc
    if isinstance(exc, ForecastFingerprintError):
        raise HTTPException(status_code=400, detail={"error": "missing_fingerprint", "message": str(exc)}) from exc
    if isinstance(exc, ForecastUntrainedModelError):
        raise HTTPException(status_code=400, detail={"error": "untrained_model", "message": str(exc)}) from exc
    if isinstance(exc, ForecastHistoryError):
        raise HTTPException(status_code=400, detail={"error": "history_error", "message": str(exc)}) from exc
    raise HTTPException(status_code=500, detail={"error": "internal_error", "message": str(exc)}) from exc
