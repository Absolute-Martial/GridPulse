"""NetworkX-based grid graph for the GridPulse digital twin."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class EdgeSpec:
    line_id: str
    from_bus: str
    to_bus: str
    capacity_mw: float
    impedance: float
    edge_type: str


LINE_SPECS: tuple[EdgeSpec, ...] = (
    EdgeSpec("line-1", "bus-1", "bus-2", 22.0, 0.12, "transmission_line"),
    EdgeSpec("line-2", "bus-2", "bus-3", 24.0, 0.11, "transmission_line"),
    EdgeSpec("line-3", "bus-3", "bus-4", 26.0, 0.10, "transmission_line"),
    EdgeSpec("line-4", "bus-4", "bus-5", 28.0, 0.09, "transmission_line"),
    EdgeSpec("line-5", "bus-5", "bus-6", 30.0, 0.10, "transmission_line"),
    EdgeSpec("line-6", "bus-6", "bus-7", 32.0, 0.12, "transmission_line"),
    EdgeSpec("line-7", "bus-7", "bus-8", 24.0, 0.13, "transmission_line"),
    EdgeSpec("line-8", "bus-8", "bus-1", 27.0, 0.14, "transmission_line"),
    EdgeSpec("line-9", "bus-2", "bus-5", 29.0, 0.07, "transmission_line"),
    EdgeSpec("line-10", "bus-3", "bus-6", 31.0, 0.08, "transmission_line"),
    EdgeSpec("trafo-1", "bus-1", "bus-4", 18.0, 0.20, "transformer"),
    EdgeSpec("trafo-2", "bus-5", "bus-8", 18.0, 0.20, "transformer"),
)

ZONE_BUS_MAP: dict[str, str] = {
    "zone-1": "bus-1",
    "zone-2": "bus-3",
    "zone-3": "bus-5",
    "zone-4": "bus-7",
    "zone-5": "bus-8",
}

GENERATOR_BUS_MAP: dict[str, str] = {
    "gen-1": "bus-1",
    "gen-2": "bus-7",
}

RENEWABLE_BUS_MAP: dict[str, str] = {
    "solar-1": "bus-3",
    "wind-1": "bus-6",
}

BATTERY_BUS_MAP: dict[str, str] = {
    "battery-1": "bus-5",
}

_GRID_GRAPH: nx.Graph | None = None


def build_default_grid() -> nx.Graph:
    global _GRID_GRAPH

    graph = nx.Graph()
    for bus_index in range(1, 9):
        graph.add_node(
            f"bus-{bus_index}",
            node_type="bus",
            voltage_pu=1.0,
            current_a=0.0,
            frequency_hz=50.0,
        )

    for zone_id, bus_id in ZONE_BUS_MAP.items():
        graph.add_node(
            zone_id,
            node_type="zone",
            attached_bus=bus_id,
            load_mw=0.0,
            generation_mw=0.0,
            renewable_generation_mw=0.0,
            temperature_c=0.0,
            humidity_percent=0.0,
        )
        graph.add_node(
            f"load-{zone_id}",
            node_type="load",
            attached_zone=zone_id,
            attached_bus=bus_id,
            load_mw=0.0,
        )

    for generator_id, bus_id in GENERATOR_BUS_MAP.items():
        graph.add_node(
            generator_id,
            node_type="generator",
            attached_bus=bus_id,
            generation_mw=0.0,
        )

    for renewable_id, bus_id in RENEWABLE_BUS_MAP.items():
        graph.add_node(
            renewable_id,
            node_type="renewable_source",
            attached_bus=bus_id,
            renewable_generation_mw=0.0,
        )

    for battery_id, bus_id in BATTERY_BUS_MAP.items():
        graph.add_node(
            battery_id,
            node_type="battery",
            attached_bus=bus_id,
            storage_soc_percent=0.0,
        )

    for spec in LINE_SPECS:
        graph.add_edge(
            spec.from_bus,
            spec.to_bus,
            line_id=spec.line_id,
            from_bus=spec.from_bus,
            to_bus=spec.to_bus,
            capacity_mw=spec.capacity_mw,
            current_flow_mw=0.0,
            impedance=spec.impedance,
            status="active",
            loading_percent=0.0,
            edge_type=spec.edge_type,
            previous_flow_mw=0.0,
        )

    graph.graph["last_updated"] = None
    _GRID_GRAPH = graph
    return graph


def update_grid_state(sensor_dataframe: pd.DataFrame) -> nx.Graph:
    graph = _ensure_graph()
    if sensor_dataframe.empty:
        return graph

    frame = sensor_dataframe.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest_snapshot = frame[frame["timestamp"] == frame["timestamp"].max()].copy()

    for _, row in latest_snapshot.iterrows():
        edge = _get_edge_by_line_id(graph, str(row["line_id"]))
        if edge is not None:
            status = "failed" if row["fault_status"] == "fault" else "active"
            current_flow_mw = 0.0 if status == "failed" else float(row["line_flow_mw"])
            capacity_mw = float(row["line_capacity_mw"])
            loading_percent = 0.0 if capacity_mw == 0 else (current_flow_mw / capacity_mw) * 100.0
            edge["capacity_mw"] = capacity_mw
            edge["current_flow_mw"] = current_flow_mw
            edge["status"] = status
            edge["loading_percent"] = round(loading_percent, 2)
            edge["previous_flow_mw"] = current_flow_mw

    for bus_id, bus_frame in latest_snapshot.groupby("bus_id"):
        bus_node = graph.nodes[str(bus_id)]
        bus_node["voltage_pu"] = round(float(bus_frame["voltage_pu"].mean()), 4)
        bus_node["current_a"] = round(float(bus_frame["current_a"].mean()), 2)
        bus_node["frequency_hz"] = round(float(bus_frame["frequency_hz"].mean()), 4)
        bus_node["load_mw"] = round(float(bus_frame["load_mw"].sum()), 3)
        bus_node["generation_mw"] = round(float(bus_frame["generation_mw"].sum()), 3)
        bus_node["renewable_generation_mw"] = round(
            float(bus_frame["renewable_generation_mw"].sum()), 3
        )

    for zone_id, zone_frame in latest_snapshot.groupby("zone_id"):
        zone_node = graph.nodes[str(zone_id)]
        zone_node["load_mw"] = round(float(zone_frame["load_mw"].sum()), 3)
        zone_node["generation_mw"] = round(float(zone_frame["generation_mw"].sum()), 3)
        zone_node["renewable_generation_mw"] = round(
            float(zone_frame["renewable_generation_mw"].sum()), 3
        )
        zone_node["temperature_c"] = round(float(zone_frame["temperature_c"].mean()), 2)
        zone_node["humidity_percent"] = round(float(zone_frame["humidity_percent"].mean()), 2)
        graph.nodes[f"load-{zone_id}"]["load_mw"] = zone_node["load_mw"]

    for generator_id, bus_id in GENERATOR_BUS_MAP.items():
        bus_generation = float(graph.nodes[bus_id].get("generation_mw", 0.0))
        renewable_generation = float(graph.nodes[bus_id].get("renewable_generation_mw", 0.0))
        graph.nodes[generator_id]["generation_mw"] = round(
            max(bus_generation - renewable_generation, 0.0),
            3,
        )

    for renewable_id, bus_id in RENEWABLE_BUS_MAP.items():
        graph.nodes[renewable_id]["renewable_generation_mw"] = round(
            float(graph.nodes[bus_id].get("renewable_generation_mw", 0.0)),
            3,
        )

    storage_soc = round(float(latest_snapshot["storage_soc_percent"].mean()), 2)
    for battery_id in BATTERY_BUS_MAP:
        graph.nodes[battery_id]["storage_soc_percent"] = storage_soc

    graph.graph["last_updated"] = latest_snapshot["timestamp"].max().isoformat()
    return graph


def get_grid_summary() -> dict[str, Any]:
    graph = _ensure_graph()
    totals = _totals_from_graph(graph)
    edges = list(graph.edges(data=True))
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "bus_count": _count_nodes(graph, "bus"),
        "zone_count": _count_nodes(graph, "zone"),
        "generator_count": _count_nodes(graph, "generator"),
        "renewable_count": _count_nodes(graph, "renewable_source"),
        "battery_count": _count_nodes(graph, "battery"),
        "load_count": _count_nodes(graph, "load"),
        "transmission_line_count": sum(1 for _, _, data in edges if data["edge_type"] == "transmission_line"),
        "transformer_count": sum(1 for _, _, data in edges if data["edge_type"] == "transformer"),
        "active_line_count": sum(1 for _, _, data in edges if data["status"] == "active"),
        "failed_line_count": sum(1 for _, _, data in edges if data["status"] == "failed"),
        "overloaded_line_count": len(get_overloaded_lines()),
        "total_load_mw": totals["total_load_mw"],
        "total_generation_mw": totals["total_generation_mw"],
        "total_renewable_generation_mw": totals["total_renewable_generation_mw"],
        "average_storage_soc_percent": totals["average_storage_soc_percent"],
        "last_updated": graph.graph.get("last_updated"),
    }


def get_overloaded_lines() -> list[dict[str, Any]]:
    graph = _ensure_graph()
    overloads: list[dict[str, Any]] = []
    for _, _, data in graph.edges(data=True):
        if data["status"] == "active" and float(data["loading_percent"]) > 90.0:
            overloads.append(_edge_payload(data))
    return sorted(overloads, key=lambda item: item["loading_percent"], reverse=True)


def inject_fault(line_id: str) -> dict[str, Any]:
    graph = _ensure_graph()
    edge = _get_edge_by_line_id(graph, line_id)
    if edge is None:
        raise KeyError(f"Unknown line_id: {line_id}")

    edge["previous_flow_mw"] = edge["current_flow_mw"]
    edge["current_flow_mw"] = 0.0
    edge["loading_percent"] = 0.0
    edge["status"] = "failed"
    graph.graph["last_updated"] = datetime.now(UTC).isoformat()
    return _edge_payload(edge)


def clear_fault(line_id: str) -> dict[str, Any]:
    graph = _ensure_graph()
    edge = _get_edge_by_line_id(graph, line_id)
    if edge is None:
        raise KeyError(f"Unknown line_id: {line_id}")

    edge["status"] = "active"
    edge["current_flow_mw"] = float(edge.get("previous_flow_mw", 0.0))
    if edge["capacity_mw"]:
        edge["loading_percent"] = round(
            (edge["current_flow_mw"] / edge["capacity_mw"]) * 100.0,
            2,
        )
    else:
        edge["loading_percent"] = 0.0
    graph.graph["last_updated"] = datetime.now(UTC).isoformat()
    return _edge_payload(edge)


def find_alternative_path(source_bus: str, target_bus: str) -> list[str]:
    graph = _ensure_graph()
    active_graph = nx.Graph()
    active_graph.add_nodes_from(
        (node, data)
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "bus"
    )

    for from_node, to_node, data in graph.edges(data=True):
        if data["status"] != "active":
            continue
        active_graph.add_edge(
            from_node,
            to_node,
            weight=float(data["impedance"]) * (1.0 + (float(data["loading_percent"]) / 100.0)),
            line_id=data["line_id"],
        )

    try:
        return nx.shortest_path(active_graph, source=source_bus, target=target_bus, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def serialize_grid_state() -> dict[str, Any]:
    graph = _ensure_graph()
    nodes = [
        {
            "node_id": node_id,
            **data,
        }
        for node_id, data in graph.nodes(data=True)
    ]
    edges = [
        _edge_payload(data)
        for _, _, data in graph.edges(data=True)
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "summary": get_grid_summary(),
    }


def _ensure_graph() -> nx.Graph:
    if _GRID_GRAPH is None:
        return build_default_grid()
    return _GRID_GRAPH


def _get_edge_by_line_id(graph: nx.Graph, line_id: str) -> dict[str, Any] | None:
    for _, _, data in graph.edges(data=True):
        if data["line_id"] == line_id:
            return data
    return None


def _count_nodes(graph: nx.Graph, node_type: str) -> int:
    return sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == node_type)


def _totals_from_graph(graph: nx.Graph) -> dict[str, float]:
    zone_nodes = [data for _, data in graph.nodes(data=True) if data.get("node_type") == "zone"]
    battery_nodes = [data for _, data in graph.nodes(data=True) if data.get("node_type") == "battery"]
    return {
        "total_load_mw": round(sum(float(node.get("load_mw", 0.0)) for node in zone_nodes), 3),
        "total_generation_mw": round(
            sum(float(node.get("generation_mw", 0.0)) for node in zone_nodes),
            3,
        ),
        "total_renewable_generation_mw": round(
            sum(float(node.get("renewable_generation_mw", 0.0)) for node in zone_nodes),
            3,
        ),
        "average_storage_soc_percent": round(
            (
                sum(float(node.get("storage_soc_percent", 0.0)) for node in battery_nodes)
                / max(len(battery_nodes), 1)
            ),
            2,
        ),
    }


def _edge_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "line_id": data["line_id"],
        "from_bus": data["from_bus"],
        "to_bus": data["to_bus"],
        "capacity_mw": float(data["capacity_mw"]),
        "current_flow_mw": float(data["current_flow_mw"]),
        "impedance": float(data["impedance"]),
        "status": data["status"],
        "loading_percent": float(data["loading_percent"]),
        "edge_type": data["edge_type"],
    }
