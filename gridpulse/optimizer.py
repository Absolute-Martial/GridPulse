"""Dispatch optimization using NetworkX min-cost flow with graceful fallback."""
from __future__ import annotations

from collections import defaultdict

import networkx as nx

from gridpulse.router import find_best_route, line_overloads
from gridpulse.schemas import DispatchPlan, ZoneState
from gridpulse.topology import active_graph, supply_ids, validate_with_pandapower, zone_ids


def generate_dispatch_plan(
    graph: nx.Graph,
    zone_states: list[ZoneState],
    faulted_nodes: list[str],
    enable_pandapower: bool = False,
) -> DispatchPlan:
    active = active_graph(graph, faulted_nodes)
    zone_lookup = {zone.zone_id: zone for zone in zone_states}
    demand_total = sum(zone.current_load_kw for zone in zone_states if zone.zone_id in active)

    source_caps: dict[str, float] = {}
    renewable_supply = 0.0
    total_supply_cap = 0.0
    for node_id in supply_ids(graph):
        if node_id in faulted_nodes or node_id not in active:
            continue
        attrs = graph.nodes[node_id]
        if attrs.get("kind") == "generator":
            factor = 0.85 if attrs.get("renewable") else 1.0
            cap = float(attrs.get("capacity_kw", 0.0)) * factor
            if attrs.get("renewable"):
                renewable_supply += cap
        else:
            cap = float(attrs.get("capacity_kw", 0.0)) * float(attrs.get("soc", 0.5))
        source_caps[node_id] = cap
        total_supply_cap += cap

    service_ratio = min(1.0, total_supply_cap / demand_total) if demand_total else 1.0
    degraded = service_ratio < 0.999 or bool(faulted_nodes)

    served_demand: dict[str, float] = {}
    curtailed_zone_ids: list[str] = []
    for zone in zone_states:
        target = zone.current_load_kw * service_ratio
        served_demand[zone.zone_id] = target
        if target + 1e-6 < zone.current_load_kw:
            curtailed_zone_ids.append(zone.zone_id)

    flow_graph = nx.DiGraph()
    active_nodes = set(active.nodes())
    for node_id in active_nodes:
        flow_graph.add_node(node_id, demand=0)
    for source, cap in source_caps.items():
        if source in active_nodes:
            scaled_supply = cap if total_supply_cap == 0 else cap * (sum(served_demand.values()) / total_supply_cap)
            flow_graph.nodes[source]["demand"] = -int(round(scaled_supply))
    for zone_id, demand in served_demand.items():
        if zone_id in active_nodes:
            flow_graph.nodes[zone_id]["demand"] = int(round(demand))

    for left, right, attrs in active.edges(data=True):
        cost = int(round(float(attrs["resistance"]) * 100))
        capacity = int(round(float(attrs["capacity_kw"])))
        flow_graph.add_edge(left, right, weight=cost, capacity=capacity)
        flow_graph.add_edge(right, left, weight=cost, capacity=capacity)

    try:
        _, flows = nx.network_simplex(flow_graph)
    except Exception:
        flows = {node: {} for node in flow_graph.nodes()}

    edge_flows: defaultdict[tuple[str, str], float] = defaultdict(float)
    for src, targets in flows.items():
        for dst, value in targets.items():
            if value <= 0:
                continue
            key = tuple(sorted((src, dst)))
            edge_flows[key] += float(value)

    line_flows: list[dict] = []
    for left, right, attrs in active.edges(data=True):
        key = tuple(sorted((left, right)))
        flow_kw = edge_flows.get(key, 0.0)
        capacity_kw = float(attrs["capacity_kw"])
        utilization = (flow_kw / capacity_kw * 100.0) if capacity_kw else 0.0
        line_flows.append(
            {
                "from": left,
                "to": right,
                "flow_kw": flow_kw,
                "capacity_kw": capacity_kw,
                "utilization_pct": utilization,
            }
        )

    routes = {
        zone_id: find_best_route(graph, zone_id, faulted_nodes)
        for zone_id in zone_ids(graph)
        if zone_id in zone_lookup
    }
    overloaded = line_overloads(line_flows, threshold=90.0)
    supplied_total = sum(served_demand.values())
    renewable_share = min(1.0, renewable_supply / supplied_total) if supplied_total else 0.0
    validation = validate_with_pandapower(graph, zone_states, line_flows, enable_pandapower)
    transmission_loss_kw = sum(
        float(item["flow_kw"]) * (float(active.edges[item["from"], item["to"]]["resistance"]) * 0.01)
        for item in line_flows
    )
    overload_penalty_kw = sum(
        max(0.0, float(item["utilization_pct"]) - 100.0) / 100.0 * float(item["capacity_kw"])
        for item in line_flows
    )
    renewable_bonus_kw = supplied_total * renewable_share * 0.08
    objective_score = transmission_loss_kw + overload_penalty_kw - renewable_bonus_kw

    return DispatchPlan(
        total_demand_kw=float(round(demand_total, 2)),
        total_supplied_kw=float(round(supplied_total, 2)),
        renewable_share=float(round(renewable_share, 3)),
        max_line_utilization=float(round(max((item["utilization_pct"] for item in line_flows), default=0.0), 2)),
        transmission_loss_kw=float(round(transmission_loss_kw, 2)),
        overload_penalty_kw=float(round(overload_penalty_kw, 2)),
        renewable_bonus_kw=float(round(renewable_bonus_kw, 2)),
        objective_score=float(round(objective_score, 2)),
        degraded=degraded,
        line_flows=line_flows,
        overloaded_lines=overloaded,
        routes=routes,
        curtailed_zone_ids=curtailed_zone_ids,
        validation=validation,
    )
