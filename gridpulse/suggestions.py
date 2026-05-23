"""Suggestion engine = rules + SHAP explanations."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from gridpulse.classifier import LABELS, GridStateClassifier
from gridpulse.router import disconnected_zones
from gridpulse.schemas import DispatchPlan, Suggestion, ZoneState

_SHAP_EXPLAINER_CACHE: dict = {}


def _explainer(model):
    import shap

    key = id(model)
    if key not in _SHAP_EXPLAINER_CACHE:
        _SHAP_EXPLAINER_CACHE[key] = shap.TreeExplainer(model)
    return _SHAP_EXPLAINER_CACHE[key]


def top_shap_features(classifier: GridStateClassifier, features: pd.Series, predicted_label: str, k: int = 3) -> List[tuple[str, float]]:
    x = features[classifier.feature_columns].values.reshape(1, -1)
    shap_values = _explainer(classifier.model).shap_values(x)
    class_idx = LABELS.index(predicted_label) if predicted_label in LABELS else 0
    if isinstance(shap_values, list):
        values = shap_values[class_idx][0]
    else:
        values = np.asarray(shap_values)
        values = values[0, :, class_idx] if values.ndim == 3 else values[0]

    pairs = list(zip(classifier.feature_columns, values))
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return [(name, float(value)) for name, value in pairs[:k]]


def generate_suggestions(
    classifier: GridStateClassifier,
    features: pd.Series,
    prediction_label: str,
    probabilities: Dict[str, float],
    anomaly_score: float,
    graph,
    zone_states: list[ZoneState],
    dispatch: DispatchPlan,
    faulted_nodes: list[str],
    scenario_name: str,
) -> list[Suggestion]:
    confidence = probabilities.get(prediction_label, 0.5)
    zones_by_load = sorted(zone_states, key=lambda zone: zone.current_load_kw / zone.capacity_kw, reverse=True)
    isolated = disconnected_zones(graph, faulted_nodes)
    items: list[Suggestion] = []

    if faulted_nodes:
        items.append(
            Suggestion(
                priority="HIGH",
                title=f"Reroute around faulted node {faulted_nodes[0]}",
                why=f"Active fault at {faulted_nodes[0]} reduced routing flexibility.",
                est_impact="Preserves service continuity on alternate substation paths",
                confidence=0.96,
                category="reroute",
            )
        )
    if isolated:
        items.append(
            Suggestion(
                priority="HIGH",
                title=f"Inspect isolated load zones: {', '.join(isolated[:3])}",
                why="One or more zones have no healthy path to a supply node.",
                est_impact="Restores disconnected demand and reduces outage duration",
                confidence=0.92,
                category="inspect",
            )
        )
    if dispatch.overloaded_lines:
        top_line = dispatch.overloaded_lines[0]
        items.append(
            Suggestion(
                priority="HIGH",
                title=f"Relieve overloaded corridor {top_line['from']} -> {top_line['to']}",
                why=f"Line utilization is {top_line['utilization_pct']:.0f}% of capacity.",
                est_impact="Reduces thermal stress and lowers overload risk",
                confidence=0.88,
                category="reroute",
            )
        )
    if dispatch.objective_score > 12.0:
        items.append(
            Suggestion(
                priority="MEDIUM",
                title="Rebalance dispatch to reduce loss-cost objective",
                why=f"Current objective score is {dispatch.objective_score:.2f}, indicating inefficient routing pressure.",
                est_impact="Lowers transmission losses and overload penalties in the next control window",
                confidence=0.8,
                category="optimize",
            )
        )
    if prediction_label == "Critical":
        hottest = zones_by_load[0]
        items.append(
            Suggestion(
                priority="HIGH",
                title=f"Shift deferrable load from {hottest.zone_id}",
                why="Classifier predicts a critical grid state under current demand growth.",
                est_impact="Cuts peak demand and buys time for operators to respond",
                confidence=confidence,
                category="load_shift",
            )
        )
    elif prediction_label == "Stressed":
        items.append(
            Suggestion(
                priority="MEDIUM",
                title="Pre-dispatch battery support before the next peak window",
                why="Forecast demand is rising while stability margin is tightening.",
                est_impact="Smooths ramp-up and lowers risk of emergency shedding",
                confidence=confidence,
                category="battery",
            )
        )
    elif prediction_label == "Surplus" or scenario_name == "renewable_surplus":
        items.append(
            Suggestion(
                priority="MEDIUM",
                title="Charge storage and schedule flexible loads now",
                why="Renewable availability is above the configured surplus threshold.",
                est_impact="Improves renewable utilization and reduces future peak draw",
                confidence=max(confidence, 0.8),
                category="renewable",
            )
        )

    if anomaly_score > 0.62:
        items.append(
            Suggestion(
                priority="MEDIUM",
                title="Investigate abnormal telemetry before escalation",
                why=f"Isolation Forest anomaly score reached {anomaly_score:.2f}.",
                est_impact="Catches sensor drift, tampering, or pre-failure behavior early",
                confidence=anomaly_score,
                category="inspect",
            )
        )

    if dispatch.curtailed_zone_ids:
        items.append(
            Suggestion(
                priority="MEDIUM",
                title=f"Review curtailed zones: {', '.join(dispatch.curtailed_zone_ids[:3])}",
                why="Available supply is below full demand coverage for the current scenario.",
                est_impact="Reduces customer impact by prioritizing restoration or manual shedding",
                confidence=0.84,
                category="load_shift",
            )
        )

    if not items:
        items.append(
            Suggestion(
                priority="LOW",
                title="Grid operating within expected limits",
                why="No overloads, no active faults, and anomaly score is low.",
                est_impact="Continue normal monitoring and prepare next forecast window",
                confidence=confidence,
                category="monitor",
            )
        )

    try:
        top_features = top_shap_features(classifier, features, prediction_label)
    except Exception:
        top_features = []

    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    items.sort(key=lambda item: (priority_rank.get(item.priority, 9), -item.confidence))
    for item in items:
        item.top_features = top_features
    return items[:3]
