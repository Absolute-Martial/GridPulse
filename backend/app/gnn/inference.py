"""Inference and formatting helpers for GridPulse topology-risk predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.gnn.graph_dataset import EDGE_FEATURE_COLUMNS, NODE_FEATURE_COLUMNS, build_graph_tensors
from app.gnn.trainer import GridRiskGNNTrainer


@dataclass
class GridRiskInference:
    """Run risk inference on the current GridPulse graph state."""

    model_path: Path | None = None

    def __post_init__(self) -> None:
        self.model_path = self.model_path or get_settings().gnn_model_path
        self.trainer = GridRiskGNNTrainer(model_path=self.model_path)

    def predict_risk(self, sensor_frame: pd.DataFrame) -> list[dict[str, Any]]:
        sample = build_graph_tensors(sensor_frame, scenario_name="inference")
        model = self.trainer.build_model()
        output = model(sample.node_features, sample.edge_index, sample.edge_features)
        return self._format_risk_items(sample, output)

    def top_risk_lines(self, sensor_frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
        items = [item for item in self.predict_risk(sensor_frame) if item["affected_line"] is not None]
        return sorted(items, key=lambda item: item["risk_score"], reverse=True)[:limit]

    def top_risk_zones(self, sensor_frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
        items = [
            item
            for item in self.predict_risk(sensor_frame)
            if item["affected_node"] is not None and str(item["affected_node"]).startswith("zone-")
        ]
        return sorted(items, key=lambda item: item["risk_score"], reverse=True)[:limit]

    def _format_risk_items(self, sample, output) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        node_feature_lookup = {node_id: sample.node_features[index].tolist() for index, node_id in enumerate(sample.metadata["node_ids"])}

        for index, node_id in enumerate(sample.metadata["node_ids"]):
            if not str(node_id).startswith(("zone-", "bus-")):
                continue
            feature_values = node_feature_lookup[node_id]
            risk_type = _node_risk_type(feature_values)
            risk_score = float(output.node_risk_score[index].item())
            confidence = _confidence_from_score(risk_score)
            items.append(
                {
                    "risk_score": round(risk_score, 4),
                    "risk_type": risk_type,
                    "affected_node": node_id,
                    "affected_line": None,
                    "confidence": round(confidence, 4),
                    "features_used": _features_used_for_node(feature_values),
                }
            )

        for index, line_id in enumerate(sample.metadata["edge_ids"]):
            feature_values = sample.edge_features[index].tolist()
            line_risk = float(output.line_risk_score[index].item())
            overload_risk = float(output.overload_risk[index].item())
            risk_score = max(line_risk, overload_risk)
            items.append(
                {
                    "risk_score": round(risk_score, 4),
                    "risk_type": _edge_risk_type(feature_values, overload_risk),
                    "affected_node": None,
                    "affected_line": line_id,
                    "confidence": round(_confidence_from_score(risk_score), 4),
                    "features_used": _features_used_for_edge(feature_values),
                }
            )
        return sorted(items, key=lambda item: item["risk_score"], reverse=True)


def _confidence_from_score(risk_score: float) -> float:
    return min(0.55 + abs(risk_score - 0.5), 0.99)


def _node_risk_type(feature_values: list[float]) -> str:
    voltage = feature_values[3]
    frequency = feature_values[4]
    fault_indicator = feature_values[6]
    if fault_indicator >= 0.5:
        return "fault_adjacent"
    if voltage < 0.95 or voltage > 1.05:
        return "voltage_risk"
    if frequency < 49.5 or frequency > 50.5:
        return "frequency_risk"
    return "topology_node_risk"


def _edge_risk_type(feature_values: list[float], overload_risk: float) -> str:
    loading_percent = feature_values[2]
    failed_indicator = feature_values[4]
    if failed_indicator >= 0.5:
        return "line_fault_risk"
    if loading_percent > 90.0 or overload_risk > 0.7:
        return "overload_risk"
    return "topology_line_risk"


def _features_used_for_node(feature_values: list[float]) -> list[str]:
    features_used: list[str] = []
    for name, value in zip(NODE_FEATURE_COLUMNS, feature_values, strict=False):
        if name in {"load_mw", "generation_mw", "renewable_generation_mw"} and abs(value) > 1.0:
            features_used.append(name)
        elif name == "voltage_pu" and (value < 0.97 or value > 1.03):
            features_used.append(name)
        elif name == "frequency_hz" and abs(value - 50.0) > 0.2:
            features_used.append(name)
        elif name in {"storage_soc_percent", "fault_indicator"} and value > 0.2:
            features_used.append(name)
    return features_used or ["load_mw", "voltage_pu", "frequency_hz"]


def _features_used_for_edge(feature_values: list[float]) -> list[str]:
    features_used: list[str] = []
    for name, value in zip(EDGE_FEATURE_COLUMNS, feature_values, strict=False):
        if name == "loading_percent" and value > 80.0:
            features_used.append(name)
        elif name == "failed_indicator" and value > 0.5:
            features_used.append(name)
        elif name in {"line_flow_mw", "line_capacity_mw", "impedance"} and value > 0.0:
            features_used.append(name)
    return features_used or ["line_flow_mw", "loading_percent", "impedance"]
