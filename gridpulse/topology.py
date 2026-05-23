"""NetworkX-first topology builder with optional pandapower validation helpers."""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from gridpulse.data_loader import load_topology_spec


def load_demo_grid(path: str) -> nx.Graph:
    spec = load_topology_spec(path)
    graph = nx.Graph()
    for node in spec["nodes"]:
        graph.add_node(node["id"], **node)
    for edge in spec["edges"]:
        graph.add_edge(
            edge["source"],
            edge["target"],
            capacity_kw=float(edge["capacity_kw"]),
            resistance=float(edge["resistance"]),
        )
    return graph


def active_graph(graph: nx.Graph, faulted_nodes: Iterable[str]) -> nx.Graph:
    active = graph.copy()
    active.remove_nodes_from([node for node in faulted_nodes if node in active])
    return active


def grid_summary(graph: nx.Graph) -> dict:
    kinds = nx.get_node_attributes(graph, "kind")
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "zones": sum(1 for kind in kinds.values() if kind == "zone"),
        "substations": sum(1 for kind in kinds.values() if kind == "substation"),
        "generators": sum(1 for kind in kinds.values() if kind == "generator"),
        "batteries": sum(1 for kind in kinds.values() if kind == "battery"),
    }


def zone_ids(graph: nx.Graph) -> list[str]:
    return [node for node, attrs in graph.nodes(data=True) if attrs.get("kind") == "zone"]


def supply_ids(graph: nx.Graph) -> list[str]:
    return [node for node, attrs in graph.nodes(data=True) if attrs.get("kind") in {"generator", "battery"}]


def zone_capacity_map(graph: nx.Graph) -> dict[str, float]:
    capacities: dict[str, float] = {}
    for zone_id in zone_ids(graph):
        neighbors = list(graph.neighbors(zone_id))
        capacities[zone_id] = sum(float(graph.edges[zone_id, neighbor]["capacity_kw"]) for neighbor in neighbors)
    return capacities


def validate_with_pandapower(graph: nx.Graph, zone_states: list, dispatch, enabled: bool) -> dict | None:
    if not enabled:
        return None
    try:
        import pandapower as pp
    except Exception as exc:
        return {"enabled": True, "available": False, "error": str(exc)}

    net = pp.create_empty_network()
    bus_index: dict[str, int] = {}
    for node_id, attrs in graph.nodes(data=True):
        bus_index[node_id] = pp.create_bus(net, vn_kv=20.0, name=node_id)
        if attrs.get("kind") == "generator":
            pp.create_ext_grid(net, bus=bus_index[node_id], vm_pu=1.0)

    for left, right, attrs in graph.edges(data=True):
        pp.create_line_from_parameters(
            net,
            from_bus=bus_index[left],
            to_bus=bus_index[right],
            length_km=1.0,
            r_ohm_per_km=float(attrs["resistance"]),
            x_ohm_per_km=0.1,
            c_nf_per_km=0.0,
            max_i_ka=max(float(attrs["capacity_kw"]) / 20000.0, 0.05),
        )

    zone_lookup = {zone.zone_id: zone for zone in zone_states}
    for zone_id, state in zone_lookup.items():
        pp.create_load(net, bus=bus_index[zone_id], p_mw=state.current_load_kw / 1000.0, q_mvar=0.02)

    try:
        pp.runpp(net, init="flat")
        loss_kw = float(net.res_line["pl_mw"].sum() * 1000.0) if len(net.res_line) else 0.0
        return {"enabled": True, "available": True, "converged": True, "loss_kw": loss_kw}
    except Exception as exc:
        return {"enabled": True, "available": True, "converged": False, "error": str(exc)}
