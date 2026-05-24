"""Graph tensor creation and synthetic scenario labeling for the GridPulse GNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from app.digital_twin.grid_graph import build_default_grid, update_grid_state
from app.simulator.sensor_generator import generate_sensor_data

NODE_FEATURE_COLUMNS: tuple[str, ...] = (
    "load_mw",
    "generation_mw",
    "renewable_generation_mw",
    "voltage_pu",
    "frequency_hz",
    "storage_soc_percent",
    "fault_indicator",
)

EDGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "line_flow_mw",
    "line_capacity_mw",
    "loading_percent",
    "impedance",
    "failed_indicator",
)

SCENARIO_TYPES: tuple[str, ...] = (
    "normal",
    "high_load",
    "low_renewable",
    "line_fault",
    "voltage_anomaly",
    "frequency_anomaly",
    "overload",
)


@dataclass
class GraphTensors:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_targets: torch.Tensor
    edge_targets: torch.Tensor
    metadata: dict[str, Any]


class SyntheticGraphScenarioDataset(Dataset):
    """Small in-memory dataset of synthetic graph-risk scenarios."""

    def __init__(self, samples: list[GraphTensors]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphTensors:
        return self.samples[index]


def build_graph_tensors(
    sensor_frame: pd.DataFrame,
    scenario_name: str = "normal",
) -> GraphTensors:
    """Convert the latest digital twin state into tensors and heuristic labels."""

    graph = build_default_grid()
    update_grid_state(sensor_frame)
    latest_frame = _latest_snapshot(sensor_frame)
    latest_by_bus = latest_frame.groupby("bus_id")
    latest_by_zone = latest_frame.groupby("zone_id")
    latest_by_line = latest_frame.groupby("line_id")
    storage_soc = float(latest_frame["storage_soc_percent"].mean())

    node_ids = sorted(graph.nodes)
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    node_features: list[list[float]] = []
    node_targets: list[list[float]] = []

    faulted_lines = {
        str(row["line_id"])
        for _, row in latest_frame.iterrows()
        if str(row["fault_status"]) == "fault"
    }

    for node_id in node_ids:
        data = graph.nodes[node_id]
        node_type = str(data.get("node_type", "bus"))
        bus_id = _resolve_bus_id(node_id=node_id, node_type=node_type, node_data=data)

        bus_rows = latest_by_bus.get_group(bus_id) if bus_id in latest_by_bus.groups else pd.DataFrame()
        zone_id = _resolve_zone_id(node_id=node_id, node_type=node_type, node_data=data)
        zone_rows = latest_by_zone.get_group(zone_id) if zone_id and zone_id in latest_by_zone.groups else pd.DataFrame()

        load_mw = float(data.get("load_mw", 0.0))
        generation_mw = float(data.get("generation_mw", 0.0))
        renewable_mw = float(data.get("renewable_generation_mw", 0.0))
        if node_type == "bus" and not bus_rows.empty:
            load_mw = float(bus_rows["load_mw"].sum())
            generation_mw = float(bus_rows["generation_mw"].sum())
            renewable_mw = float(bus_rows["renewable_generation_mw"].sum())
        elif node_type in {"zone", "load"} and not zone_rows.empty:
            load_mw = float(zone_rows["load_mw"].sum())
            generation_mw = float(zone_rows["generation_mw"].sum())
            renewable_mw = float(zone_rows["renewable_generation_mw"].sum())

        voltage_pu = float(data.get("voltage_pu", 1.0))
        frequency_hz = float(data.get("frequency_hz", 50.0))
        if not bus_rows.empty:
            voltage_pu = float(bus_rows["voltage_pu"].mean())
            frequency_hz = float(bus_rows["frequency_hz"].mean())

        fault_indicator = 1.0 if _node_fault_indicator(graph, node_id, faulted_lines) else 0.0
        storage_feature = storage_soc if node_type == "battery" else storage_soc * 0.5
        feature_row = [
            load_mw,
            generation_mw,
            renewable_mw,
            voltage_pu,
            frequency_hz,
            storage_feature,
            fault_indicator,
        ]
        node_features.append(feature_row)
        node_targets.append(
            [
                _compute_node_risk(feature_row),
                _compute_restoration_priority(feature_row, node_type=node_type),
            ]
        )

    edge_pairs: list[list[int]] = []
    edge_features: list[list[float]] = []
    edge_targets: list[list[float]] = []
    edge_ids: list[str] = []

    for from_node, to_node, data in graph.edges(data=True):
        line_id = str(data["line_id"])
        edge_pairs.append([node_index[from_node], node_index[to_node]])
        line_rows = latest_by_line.get_group(line_id) if line_id in latest_by_line.groups else pd.DataFrame()
        line_flow_mw = float(data.get("current_flow_mw", 0.0))
        line_capacity_mw = float(data.get("capacity_mw", 1.0))
        loading_percent = float(data.get("loading_percent", 0.0))
        failed_indicator = 1.0 if str(data.get("status", "active")) == "failed" else 0.0
        if not line_rows.empty:
            line_flow_mw = float(line_rows["line_flow_mw"].mean())
            line_capacity_mw = float(line_rows["line_capacity_mw"].mean())
            loading_percent = float((line_flow_mw / max(line_capacity_mw, 1e-6)) * 100.0)
            failed_indicator = 1.0 if str(line_rows["fault_status"].iloc[0]) == "fault" else failed_indicator

        feature_row = [
            line_flow_mw,
            line_capacity_mw,
            loading_percent,
            float(data.get("impedance", 0.0)),
            failed_indicator,
        ]
        edge_ids.append(line_id)
        edge_features.append(feature_row)
        edge_targets.append(
            [
                _compute_edge_risk(feature_row),
                _compute_overload_risk(feature_row),
            ]
        )

    return GraphTensors(
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
        edge_features=torch.tensor(edge_features, dtype=torch.float32),
        node_targets=torch.tensor(node_targets, dtype=torch.float32),
        edge_targets=torch.tensor(edge_targets, dtype=torch.float32),
        metadata={
            "scenario_name": scenario_name,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "node_feature_columns": list(NODE_FEATURE_COLUMNS),
            "edge_feature_columns": list(EDGE_FEATURE_COLUMNS),
        },
    )


def generate_scenario_dataset(
    base_sensor_frame: pd.DataFrame | None = None,
    scenarios_per_type: int = 2,
) -> SyntheticGraphScenarioDataset:
    """Create a compact labeled dataset from synthetic perturbation scenarios."""

    samples: list[GraphTensors] = []
    for scenario_name in SCENARIO_TYPES:
        for scenario_index in range(scenarios_per_type):
            if base_sensor_frame is None:
                frame = generate_sensor_data(hours=72, interval_minutes=60, seed=7 + scenario_index)
            else:
                frame = base_sensor_frame.copy()
            scenario_frame = apply_scenario(frame, scenario_name, scenario_index)
            samples.append(build_graph_tensors(scenario_frame, scenario_name=scenario_name))
    return SyntheticGraphScenarioDataset(samples)


def apply_scenario(sensor_frame: pd.DataFrame, scenario_name: str, scenario_index: int = 0) -> pd.DataFrame:
    """Mutate the latest sensor snapshot to create a specific risk scenario."""

    frame = sensor_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = frame["timestamp"].max()
    latest_mask = frame["timestamp"] == latest_timestamp

    if scenario_name == "normal":
        return frame
    if scenario_name == "high_load":
        frame.loc[latest_mask, "load_mw"] *= 1.35
        frame.loc[latest_mask, "line_flow_mw"] *= 1.30
        return frame
    if scenario_name == "low_renewable":
        frame.loc[latest_mask, "renewable_generation_mw"] *= 0.20
        frame.loc[latest_mask, "generation_mw"] *= 0.80
        return frame
    if scenario_name == "line_fault":
        target_line = _scenario_line_id(scenario_index)
        line_mask = latest_mask & (frame["line_id"] == target_line)
        frame.loc[line_mask, "fault_status"] = "fault"
        frame.loc[line_mask, "line_flow_mw"] = 0.0
        return frame
    if scenario_name == "voltage_anomaly":
        target_bus = _scenario_bus_id(scenario_index)
        bus_mask = latest_mask & (frame["bus_id"] == target_bus)
        frame.loc[bus_mask, "voltage_pu"] = 1.10 if scenario_index % 2 == 0 else 0.90
        return frame
    if scenario_name == "frequency_anomaly":
        target_bus = _scenario_bus_id(scenario_index + 1)
        bus_mask = latest_mask & (frame["bus_id"] == target_bus)
        frame.loc[bus_mask, "frequency_hz"] = 49.2 if scenario_index % 2 == 0 else 50.8
        return frame
    if scenario_name == "overload":
        target_line = _scenario_line_id(scenario_index + 1)
        line_mask = latest_mask & (frame["line_id"] == target_line)
        frame.loc[line_mask, "line_flow_mw"] = frame.loc[line_mask, "line_capacity_mw"] * 1.25
        return frame
    raise KeyError(f"Unsupported scenario_name: {scenario_name}")


def _latest_snapshot(sensor_frame: pd.DataFrame) -> pd.DataFrame:
    frame = sensor_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = frame["timestamp"].max()
    return frame[frame["timestamp"] == latest_timestamp].copy()


def _resolve_bus_id(node_id: str, node_type: str, node_data: dict[str, Any]) -> str:
    if node_type == "bus":
        return node_id
    if node_type == "zone":
        return str(node_data["attached_bus"])
    if node_type in {"load", "generator", "renewable_source", "battery"}:
        return str(node_data["attached_bus"])
    return "bus-1"


def _resolve_zone_id(node_id: str, node_type: str, node_data: dict[str, Any]) -> str | None:
    if node_type == "zone":
        return node_id
    if node_type == "load":
        return str(node_data.get("attached_zone"))
    if node_type in {"bus", "generator", "renewable_source", "battery"}:
        suffix = _extract_numeric_suffix(_resolve_bus_id(node_id, node_type, node_data))
        return f"zone-{min(max(suffix, 1), 5)}"
    return None


def _extract_numeric_suffix(identifier: str) -> int:
    digits = "".join(character for character in identifier if character.isdigit())
    return int(digits) if digits else 1


def _node_fault_indicator(graph, node_id: str, faulted_lines: set[str]) -> bool:
    for _, _, edge_data in graph.edges(node_id, data=True):
        if str(edge_data["line_id"]) in faulted_lines or str(edge_data.get("status")) == "failed":
            return True
    return False


def _compute_node_risk(feature_row: list[float]) -> float:
    load_mw, generation_mw, renewable_mw, voltage_pu, frequency_hz, storage_soc_percent, fault_indicator = feature_row
    load_ratio = min(load_mw / 20.0, 1.0)
    generation_gap = max(load_mw - generation_mw - renewable_mw, 0.0)
    generation_gap_ratio = min(generation_gap / max(load_mw, 1.0), 1.0)
    voltage_deviation = min(abs(voltage_pu - 1.0) / 0.1, 1.0)
    frequency_deviation = min(abs(frequency_hz - 50.0) / 1.0, 1.0)
    low_storage = 1.0 - min(max(storage_soc_percent / 100.0, 0.0), 1.0)
    risk = (
        0.22 * load_ratio
        + 0.18 * generation_gap_ratio
        + 0.18 * voltage_deviation
        + 0.18 * frequency_deviation
        + 0.12 * low_storage
        + 0.12 * fault_indicator
    )
    return float(min(max(risk, 0.0), 1.0))


def _compute_restoration_priority(feature_row: list[float], node_type: str) -> float:
    load_mw = feature_row[0]
    fault_indicator = feature_row[-1]
    base_priority = min(load_mw / 20.0, 1.0)
    if node_type in {"zone", "load", "bus"}:
        base_priority += 0.2
    priority = 0.5 * base_priority + 0.5 * fault_indicator
    return float(min(max(priority, 0.0), 1.0))


def _compute_edge_risk(feature_row: list[float]) -> float:
    line_flow_mw, line_capacity_mw, loading_percent, impedance, failed_indicator = feature_row
    capacity_ratio = min(line_flow_mw / max(line_capacity_mw, 1.0), 1.5) / 1.5
    impedance_ratio = min(impedance / 0.2, 1.0)
    risk = 0.5 * capacity_ratio + 0.2 * min(loading_percent / 100.0, 1.5) / 1.5 + 0.15 * impedance_ratio + 0.15 * failed_indicator
    return float(min(max(risk, 0.0), 1.0))


def _compute_overload_risk(feature_row: list[float]) -> float:
    loading_percent = feature_row[2]
    failed_indicator = feature_row[4]
    if failed_indicator >= 1.0:
        return 1.0
    return float(min(max((loading_percent - 70.0) / 40.0, 0.0), 1.0))


def _scenario_line_id(scenario_index: int) -> str:
    line_ids = ("line-2", "line-4", "line-6", "line-8", "line-10")
    return line_ids[scenario_index % len(line_ids)]


def _scenario_bus_id(scenario_index: int) -> str:
    bus_ids = ("bus-2", "bus-3", "bus-5", "bus-6", "bus-8")
    return bus_ids[scenario_index % len(bus_ids)]
