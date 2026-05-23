"""Routing helpers for NetworkX-first GridPulse."""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from gridpulse.topology import active_graph, supply_ids, zone_ids


def find_best_route(graph: nx.Graph, zone_id: str, faulted_nodes: Iterable[str]) -> list[str]:
    active = active_graph(graph, faulted_nodes)
    sources = [node for node in supply_ids(active) if node in active]
    if zone_id not in active or not sources:
        return []

    best_path: list[str] = []
    best_cost = float("inf")
    for source in sources:
        try:
            path = nx.shortest_path(active, source=source, target=zone_id, weight="resistance")
            cost = nx.path_weight(active, path, weight="resistance")
            if cost < best_cost:
                best_path = path
                best_cost = cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return best_path


def disconnected_zones(graph: nx.Graph, faulted_nodes: Iterable[str]) -> list[str]:
    active = active_graph(graph, faulted_nodes)
    sources = [node for node in supply_ids(active) if node in active]
    disconnected: list[str] = []
    for zone_id in zone_ids(graph):
        if zone_id not in active:
            disconnected.append(zone_id)
            continue
        if not any(nx.has_path(active, source, zone_id) for source in sources):
            disconnected.append(zone_id)
    return disconnected


def line_overloads(line_flows: list[dict], threshold: float = 90.0) -> list[dict]:
    overloaded = [flow for flow in line_flows if flow["utilization_pct"] >= threshold]
    overloaded.sort(key=lambda item: item["utilization_pct"], reverse=True)
    return overloaded
