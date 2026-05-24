"""Forecast explanation helpers for tree-based models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

from app.forecasting.features import TARGET_MODEL_FEATURE_COLUMNS, build_target_feature_frame
from app.forecasting.response_formatter import build_forecast_response
from app.forecasting.tree_forecaster import TreeForecaster


@dataclass
class _PeakWrapper(RegressorMixin, BaseEstimator):
    model: Any

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "_PeakWrapper":
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.max(self.model.predict(x), axis=1)


def explain_tree_forecast(
    *,
    forecaster: TreeForecaster,
    dataframe: pd.DataFrame,
    entity_type: str,
    entity_id: str,
    horizon: str,
    target: str = "peak",
) -> dict[str, Any]:
    """Explain a tree-based forecast with SHAP or permutation fallback."""

    feature_frame = build_target_feature_frame(
        dataframe,
        entity_type=entity_type,
        entity_id=entity_id,
        lookback_steps=480,
        horizon_steps=1,
    )
    latest_row = feature_frame.iloc[[-1]]
    raw_forecast = forecaster.predict(
        dataframe,
        entity_type=entity_type,
        entity_id=entity_id,
        horizon=horizon,
    )
    forecast = build_forecast_response(
        entity_type=entity_type,
        entity_id=entity_id,
        horizon=horizon,
        latest_timestamp=raw_forecast["latest_input_timestamp"],
        slot_predictions=raw_forecast["slot_predictions"],
        confidence=raw_forecast.get("summary", {}).get("confidence", "medium"),
        model_name=forecaster.model_name,
    )

    x_latest = latest_row.loc[:, list(TARGET_MODEL_FEATURE_COLUMNS)]
    model = forecaster._artifact_model()

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_latest)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[0]).reshape(-1)
        else:
            values = np.asarray(shap_values).reshape(-1)
        method = "shap"
        ranked = _rank_features(values, x_latest.iloc[0].to_dict())
    except Exception:
        dataset = forecaster._build_training_dataset(  # noqa: SLF001
            dataframe,
            entity_type=entity_type,
            entity_id=entity_id,
            horizon_steps=forecast["horizon_steps"],
        )
        x_eval = dataset.loc[:, list(TARGET_MODEL_FEATURE_COLUMNS)]
        y_peak = np.asarray([max(sequence) for sequence in dataset["target_sequence"]], dtype=float)
        result = permutation_importance(
            _PeakWrapper(model),
            x_eval,
            y_peak,
            n_repeats=5,
            random_state=7,
            scoring="neg_mean_absolute_error",
        )
        values = np.asarray(result.importances_mean)
        method = "permutation_importance"
        ranked = _rank_features(values, x_latest.iloc[0].to_dict())

    top_features = ranked[:5]
    feature_names = ", ".join(item["feature"] for item in top_features[:3])
    plain_language_explanation = (
        f"{entity_type.capitalize()} {entity_id} is forecasted to peak at "
        f"{forecast['summary']['peak_time']} because {feature_names} are the strongest drivers "
        f"in the current 15-minute forecast window."
    )
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "horizon": horizon,
        "method": method,
        "target": target,
        "top_features": top_features,
        "plain_language_explanation": plain_language_explanation,
        "forecast_summary": forecast["summary"],
    }


def _rank_features(values: np.ndarray, latest_values: dict[str, Any]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for feature_name, importance in sorted(
        zip(TARGET_MODEL_FEATURE_COLUMNS, values, strict=False),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    ):
        ranked.append(
            {
                "feature": feature_name,
                "importance": round(float(importance), 6),
                "feature_value": round(float(latest_values[feature_name]), 6),
            }
        )
    return ranked
