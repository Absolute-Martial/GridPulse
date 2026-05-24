"""Canonical response formatting for 15-minute forecasting outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.forecasting.schema import normalize_horizon


def build_aggregated_summary(
    slot_predictions: list[dict[str, Any]],
    slots_per_group: int = 4,
) -> list[dict[str, Any]]:
    """Aggregate slot predictions into rolling hourly reporting groups."""

    if not slot_predictions:
        return []

    groups: list[dict[str, Any]] = []
    for start in range(0, len(slot_predictions), slots_per_group):
        chunk = slot_predictions[start : start + slots_per_group]
        predicted = [float(item["predicted_load_kw"]) for item in chunk]
        peak_index = max(range(len(predicted)), key=predicted.__getitem__)
        groups.append(
            {
                "hour_start": chunk[0]["timestamp"],
                "mean_load_kw": round(sum(predicted) / len(predicted), 3),
                "peak_load_kw": round(max(predicted), 3),
                "peak_time": chunk[peak_index]["timestamp"],
                "total_energy_kwh": round(sum(value * 0.25 for value in predicted), 3),
            }
        )
    return groups


def build_forecast_response(
    *,
    entity_type: str,
    entity_id: str,
    horizon: str,
    latest_timestamp: str,
    slot_predictions: list[dict[str, Any]],
    confidence: str,
    model_name: str,
    lookback_days: int = 5,
    lookback_steps: int = 480,
    base_resolution: str = "15min",
) -> dict[str, Any]:
    """Build the canonical API-ready forecasting payload."""

    normalized_horizon, horizon_steps = normalize_horizon(horizon)
    aggregated_summary = build_aggregated_summary(slot_predictions)
    predicted_values = [float(item["predicted_load_kw"]) for item in slot_predictions]
    peak_index = max(range(len(predicted_values)), key=predicted_values.__getitem__)

    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "latest_input_timestamp": latest_timestamp,
        "lookback_days": lookback_days,
        "lookback_steps": lookback_steps,
        "base_resolution": base_resolution,
        "horizon": normalized_horizon,
        "horizon_steps": horizon_steps,
        "model_name": model_name,
        "slot_predictions": slot_predictions,
        "aggregated_summary": aggregated_summary,
        "summary": {
            "mean_load_kw": round(sum(predicted_values) / len(predicted_values), 3),
            "peak_load_kw": round(max(predicted_values), 3),
            "peak_time": slot_predictions[peak_index]["timestamp"],
            "total_energy_kwh": round(sum(value * 0.25 for value in predicted_values), 3),
            "confidence": confidence,
        },
    }
