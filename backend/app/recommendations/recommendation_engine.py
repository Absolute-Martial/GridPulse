"""Rule-based recommendation engine for GridPulse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.anomaly.anomaly_detector import (
    AnomalyModelNotFoundError,
    GridAnomalyDetector,
)
from app.core.config import get_settings
from app.digital_twin.grid_graph import (
    find_alternative_path,
    get_grid_summary,
    get_overloaded_lines,
    serialize_grid_state,
    update_grid_state,
)
from app.forecasting.load_forecaster import (
    ForecastModelNotFoundError,
    LoadForecaster,
)

PRIORITY_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


@dataclass
class GridRecommendationEngine:
    """Combine grid signals into ranked operator recommendations."""

    forecast_model_path: Path | None = None
    anomaly_model_path: Path | None = None

    def __post_init__(self) -> None:
        settings = get_settings()
        self.forecast_model_path = self.forecast_model_path or settings.forecast_model_path
        self.anomaly_model_path = self.anomaly_model_path or settings.anomaly_model_path

    def generate_recommendations(self, sensor_frame: pd.DataFrame) -> list[dict[str, Any]]:
        frame = sensor_frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        update_grid_state(frame)

        grid_state = serialize_grid_state()
        grid_summary = get_grid_summary()
        overloads = get_overloaded_lines()
        latest_snapshot = frame[frame["timestamp"] == frame["timestamp"].max()].copy()
        zone_current_loads = (
            latest_snapshot.groupby("zone_id")["load_mw"].sum().to_dict()
        )
        zone_renewables = (
            latest_snapshot.groupby("zone_id")["renewable_generation_mw"].sum().to_dict()
        )
        line_zone_map = (
            latest_snapshot.groupby("line_id")["zone_id"].first().to_dict()
        )

        forecasts = self._ensure_forecasts(frame)
        anomalies = self._ensure_anomalies(frame)

        recommendations: list[dict[str, Any]] = []
        recommendation_counter = 0

        failed_lines = [
            edge for edge in grid_state["edges"] if edge["status"] == "failed"
        ]
        for edge in failed_lines:
            affected_zone = str(line_zone_map.get(edge["line_id"], "unspecified"))
            recommendation_counter += 1
            recommendations.append(
                self._build_recommendation(
                    recommendation_counter=recommendation_counter,
                    action="isolate_fault",
                    priority="CRITICAL",
                    reason=f"Active fault is present on {edge['line_id']}. Isolation is required to protect adjacent equipment.",
                    affected_zone=affected_zone,
                    affected_line=edge["line_id"],
                    expected_impact="Prevents fault propagation and stabilizes the affected corridor.",
                    confidence_score=0.98,
                    validation_status="rule_validated",
                )
            )

            alternative_path = find_alternative_path(edge["from_bus"], edge["to_bus"])
            if alternative_path:
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="reroute_power",
                        priority="HIGH",
                        reason=(
                            f"An alternative electrical path exists around {edge['line_id']}: "
                            f"{' -> '.join(alternative_path)}."
                        ),
                        affected_zone=affected_zone,
                        affected_line=edge["line_id"],
                        expected_impact="Maintains service continuity while the failed line remains isolated.",
                        confidence_score=0.92,
                        validation_status="rule_validated",
                    )
                )

        for overload in overloads:
            affected_zone = str(line_zone_map.get(overload["line_id"], "unspecified"))
            recommendation_counter += 1
            recommendations.append(
                self._build_recommendation(
                    recommendation_counter=recommendation_counter,
                    action="reroute_power",
                    priority="HIGH",
                    reason=(
                        f"{overload['line_id']} is loaded at {overload['loading_percent']:.1f}% of capacity."
                    ),
                    affected_zone=affected_zone,
                    affected_line=overload["line_id"],
                    expected_impact="Reduces thermal stress on the overloaded line and lowers outage risk.",
                    confidence_score=0.91,
                    validation_status="rule_validated",
                )
            )

            recommendation_counter += 1
            recommendations.append(
                self._build_recommendation(
                    recommendation_counter=recommendation_counter,
                    action="reduce_load",
                    priority="HIGH",
                    reason=f"Demand in {affected_zone} is contributing to the overload on {overload['line_id']}.",
                    affected_zone=affected_zone,
                    affected_line=overload["line_id"],
                    expected_impact="Immediate demand reduction lowers line loading and avoids protective trips.",
                    confidence_score=0.88,
                    validation_status="needs_operator_review",
                )
            )

        battery_soc = float(grid_summary["average_storage_soc_percent"])
        renewable_ratio = 0.0
        if float(grid_summary["total_generation_mw"]) > 0:
            renewable_ratio = (
                float(grid_summary["total_renewable_generation_mw"])
                / float(grid_summary["total_generation_mw"])
            )

        for forecast in forecasts:
            zone_id = str(forecast["zone_id"])
            current_load = float(zone_current_loads.get(zone_id, 0.0))
            peak_load = float(forecast["peak_load_estimate_mw"])
            if current_load <= 0.0:
                continue

            if peak_load >= current_load * 1.12 and battery_soc >= 45.0:
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="use_storage",
                        priority="HIGH",
                        reason=(
                            f"Forecasted peak demand in {zone_id} rises to {peak_load:.2f} MW while storage state of charge is {battery_soc:.1f}%."
                        ),
                        affected_zone=zone_id,
                        affected_line=None,
                        expected_impact="Battery discharge can shave the peak and reduce dispatch stress.",
                        confidence_score=0.86,
                        validation_status="dispatch_candidate",
                    )
                )
            elif peak_load >= current_load * 1.12:
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="shift_load",
                        priority="MEDIUM",
                        reason=(
                            f"Forecasted demand in {zone_id} rises from {current_load:.2f} MW to {peak_load:.2f} MW without enough storage support."
                        ),
                        affected_zone=zone_id,
                        affected_line=None,
                        expected_impact="Deferring flexible demand flattens the upcoming peak window.",
                        confidence_score=0.79,
                        validation_status="needs_operator_review",
                    )
                )

            if peak_load >= current_load * 1.25 and battery_soc < 30.0:
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="reduce_load",
                        priority="HIGH",
                        reason=(
                            f"Projected peak in {zone_id} materially exceeds current demand while storage support is limited."
                        ),
                        affected_zone=zone_id,
                        affected_line=None,
                        expected_impact="Targeted curtailment lowers the risk of overloads during the forecast peak.",
                        confidence_score=0.81,
                        validation_status="needs_operator_review",
                    )
                )

            if renewable_ratio > 0.55 and zone_renewables.get(zone_id, 0.0) > 0.0 and peak_load < current_load * 0.95:
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="curtail_generation",
                        priority="LOW",
                        reason=(
                            f"Renewable contribution is high and forecasted demand in {zone_id} softens relative to current load."
                        ),
                        affected_zone=zone_id,
                        affected_line=None,
                        expected_impact="Controlled curtailment can reduce over-voltage risk and preserve stability margins.",
                        confidence_score=0.68,
                        validation_status="needs_operator_review",
                    )
                )

        for anomaly in anomalies:
            anomaly_type = str(anomaly["anomaly_type"])
            affected_zone = str(anomaly["affected_zone"])
            confidence = min(0.6 + (float(anomaly["anomaly_score"]) * 0.1), 0.99)

            if anomaly_type == "possible sensor fault":
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="inspect_sensor",
                        priority="HIGH",
                        reason=anomaly["explanation_text"],
                        affected_zone=affected_zone,
                        affected_line=self._extract_line_id(anomaly["sensor_id"]),
                        expected_impact="Field inspection can confirm whether the issue is instrumentation or a real grid event.",
                        confidence_score=confidence,
                        validation_status="awaiting_field_check",
                    )
                )
            elif anomaly_type == "frequency anomaly" and battery_soc >= 25.0:
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="use_storage",
                        priority="HIGH",
                        reason=anomaly["explanation_text"],
                        affected_zone=affected_zone,
                        affected_line=None,
                        expected_impact="Fast battery support can help arrest the local frequency deviation.",
                        confidence_score=confidence,
                        validation_status="dispatch_candidate",
                    )
                )
            elif anomaly_type == "load spike":
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="shift_load",
                        priority="MEDIUM",
                        reason=anomaly["explanation_text"],
                        affected_zone=affected_zone,
                        affected_line=None,
                        expected_impact="Moving non-critical load away from the spike window can restore margin.",
                        confidence_score=confidence,
                        validation_status="needs_operator_review",
                    )
                )
            elif anomaly_type == "line overload":
                recommendation_counter += 1
                recommendations.append(
                    self._build_recommendation(
                        recommendation_counter=recommendation_counter,
                        action="reroute_power",
                        priority="HIGH",
                        reason=anomaly["explanation_text"],
                        affected_zone=affected_zone,
                        affected_line=self._extract_line_id(anomaly["sensor_id"]),
                        expected_impact="Rebalancing flow reduces overload exposure on the affected corridor.",
                        confidence_score=confidence,
                        validation_status="rule_validated",
                    )
                )

        deduped = self._dedupe_recommendations(recommendations)
        deduped.sort(
            key=lambda item: (
                PRIORITY_ORDER[item["priority"]],
                -float(item["confidence_score"]),
                item["recommendation_id"],
            )
        )
        return deduped

    def _ensure_forecasts(self, sensor_frame: pd.DataFrame) -> list[dict[str, Any]]:
        forecaster = LoadForecaster(model_path=self.forecast_model_path)
        try:
            forecaster.load_artifact()
        except ForecastModelNotFoundError:
            forecaster.train(sensor_frame)

        zone_ids = sorted(set(sensor_frame["zone_id"].astype(str)))
        return [
            forecaster.run(zone_id=zone_id, horizon=24, history_frame=sensor_frame)
            for zone_id in zone_ids
        ]

    def _ensure_anomalies(self, sensor_frame: pd.DataFrame) -> list[dict[str, Any]]:
        detector = GridAnomalyDetector(model_path=self.anomaly_model_path)
        try:
            return detector.detect_latest(sensor_frame)
        except AnomalyModelNotFoundError:
            detector.train(sensor_frame)
            return detector.detect_latest(sensor_frame)

    def _build_recommendation(
        self,
        recommendation_counter: int,
        action: str,
        priority: str,
        reason: str,
        affected_zone: str | None,
        affected_line: str | None,
        expected_impact: str,
        confidence_score: float,
        validation_status: str,
    ) -> dict[str, Any]:
        suffix = affected_line or affected_zone or "grid"
        return {
            "recommendation_id": f"rec-{recommendation_counter:03d}-{action}-{suffix}",
            "priority": priority,
            "action": action,
            "reason": reason,
            "affected_zone": affected_zone,
            "affected_line": affected_line,
            "expected_impact": expected_impact,
            "confidence_score": round(float(confidence_score), 3),
            "validation_status": validation_status,
        }

    def _dedupe_recommendations(
        self,
        recommendations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deduped: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
        for recommendation in recommendations:
            key = (
                recommendation["action"],
                recommendation["affected_zone"],
                recommendation["affected_line"],
            )
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = recommendation
                continue

            current_rank = PRIORITY_ORDER[recommendation["priority"]]
            existing_rank = PRIORITY_ORDER[existing["priority"]]
            if current_rank < existing_rank:
                deduped[key] = recommendation
            elif current_rank == existing_rank and recommendation["confidence_score"] > existing["confidence_score"]:
                deduped[key] = recommendation
        return list(deduped.values())

    def _extract_line_id(self, sensor_id: str) -> str | None:
        parts = str(sensor_id).split("-")
        if len(parts) >= 3 and parts[1] == "line":
            return f"{parts[1]}-{parts[2]}"
        return None
