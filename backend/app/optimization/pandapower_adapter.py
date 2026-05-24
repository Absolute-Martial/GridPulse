"""Adapter layer from the NetworkX digital twin to pandapower."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import pandas as pd

try:
    import pandapower as pp
except ModuleNotFoundError:
    pp = None

from app.digital_twin.grid_graph import BATTERY_BUS_MAP, GENERATOR_BUS_MAP, RENEWABLE_BUS_MAP, ZONE_BUS_MAP

DEFAULT_VN_KV = 20.0
DEFAULT_BUS_VM_MIN = 0.95
DEFAULT_BUS_VM_MAX = 1.05


def _require_pandapower() -> None:
    if pp is None:
        raise ImportError("pandapower is required for GridPulse optimization validation but is not installed.")


def build_pandapower_network(
    grid_graph: nx.Graph,
    sensor_dataframe: pd.DataFrame,
) -> tuple[Any, dict[str, Any]]:
    """Build a simplified pandapower network from the active grid state."""

    _require_pandapower()

    frame = sensor_dataframe.copy()
    if frame.empty:
        raise ValueError("Sensor dataframe is empty.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest_snapshot = frame[frame["timestamp"] == frame["timestamp"].max()].copy()

    net = pp.create_empty_network(sn_mva=100.0)
    metadata: dict[str, Any] = {
        "bus_map": {},
        "line_map": {},
        "load_map": {},
        "generator_map": {},
        "renewable_map": {},
        "storage_map": {},
        "ext_grid_index": None,
        "snapshot_timestamp": latest_snapshot["timestamp"].max().isoformat(),
        "measured_bus_voltage": {},
        "measured_line_loading": {},
        "requested_load_mw": {},
    }

    bus_voltage = latest_snapshot.groupby("bus_id")["voltage_pu"].mean().to_dict()
    for node_id, data in grid_graph.nodes(data=True):
        if data.get("node_type") != "bus":
            continue
        bus_index = pp.create_bus(
            net,
            vn_kv=DEFAULT_VN_KV,
            name=node_id,
            min_vm_pu=DEFAULT_BUS_VM_MIN,
            max_vm_pu=DEFAULT_BUS_VM_MAX,
        )
        metadata["bus_map"][node_id] = int(bus_index)
        metadata["measured_bus_voltage"][node_id] = float(bus_voltage.get(node_id, data.get("voltage_pu", 1.0)))

    slack_bus = metadata["bus_map"]["bus-1"]
    ext_grid_index = pp.create_ext_grid(
        net,
        bus=slack_bus,
        vm_pu=float(metadata["measured_bus_voltage"].get("bus-1", 1.0)),
        name="grid-supply",
        min_p_mw=0.0,
        max_p_mw=max(float(latest_snapshot["load_mw"].sum()) * 1.5, 50.0),
        min_q_mvar=-50.0,
        max_q_mvar=50.0,
    )
    metadata["ext_grid_index"] = int(ext_grid_index)

    zone_loads = latest_snapshot.groupby("zone_id")["load_mw"].sum().to_dict()
    for zone_id, bus_id in ZONE_BUS_MAP.items():
        p_mw = float(zone_loads.get(zone_id, 0.0))
        load_index = pp.create_load(
            net,
            bus=metadata["bus_map"][bus_id],
            p_mw=p_mw,
            q_mvar=0.0,
            name=zone_id,
            min_p_mw=0.0,
            max_p_mw=max(p_mw * 1.2, p_mw + 1.0),
            min_q_mvar=-5.0,
            max_q_mvar=5.0,
            controllable=False,
        )
        metadata["load_map"][zone_id] = int(load_index)
        metadata["requested_load_mw"][zone_id] = p_mw

    bus_generation = latest_snapshot.groupby("bus_id")["generation_mw"].sum().to_dict()
    bus_renewable = latest_snapshot.groupby("bus_id")["renewable_generation_mw"].sum().to_dict()

    for generator_id, bus_id in GENERATOR_BUS_MAP.items():
        p_mw = max(float(bus_generation.get(bus_id, 0.0)) - float(bus_renewable.get(bus_id, 0.0)), 0.0)
        gen_index = pp.create_gen(
            net,
            bus=metadata["bus_map"][bus_id],
            p_mw=p_mw,
            vm_pu=float(metadata["measured_bus_voltage"].get(bus_id, 1.0)),
            name=generator_id,
            min_p_mw=0.0,
            max_p_mw=max(p_mw * 1.5, 6.0),
            min_q_mvar=-10.0,
            max_q_mvar=10.0,
            controllable=True,
        )
        metadata["generator_map"][generator_id] = int(gen_index)

    for renewable_id, bus_id in RENEWABLE_BUS_MAP.items():
        p_mw = max(float(bus_renewable.get(bus_id, 0.0)), 0.0)
        sgen_index = pp.create_sgen(
            net,
            bus=metadata["bus_map"][bus_id],
            p_mw=p_mw,
            q_mvar=0.0,
            name=renewable_id,
            min_p_mw=0.0,
            max_p_mw=max(p_mw * 1.15, 1.0),
            min_q_mvar=-5.0,
            max_q_mvar=5.0,
            controllable=True,
        )
        metadata["renewable_map"][renewable_id] = int(sgen_index)

    storage_soc = float(latest_snapshot["storage_soc_percent"].mean())
    storage_power_limit = max(storage_soc / 20.0, 1.0)
    for storage_id, bus_id in BATTERY_BUS_MAP.items():
        storage_index = pp.create_storage(
            net,
            bus=metadata["bus_map"][bus_id],
            p_mw=0.0,
            max_e_mwh=10.0,
            soc_percent=storage_soc,
            min_p_mw=-storage_power_limit,
            max_p_mw=storage_power_limit,
            min_q_mvar=-5.0,
            max_q_mvar=5.0,
            name=storage_id,
            controllable=True,
        )
        metadata["storage_map"][storage_id] = int(storage_index)

    line_snapshot = latest_snapshot.groupby("line_id", as_index=False).agg(
        line_flow_mw=("line_flow_mw", "mean"),
        line_capacity_mw=("line_capacity_mw", "mean"),
        fault_status=("fault_status", "first"),
    )
    line_state = {
        str(row["line_id"]): row
        for _, row in line_snapshot.iterrows()
    }

    for from_bus, to_bus, data in grid_graph.edges(data=True):
        line_id = str(data["line_id"])
        state_row = line_state.get(line_id)
        capacity_mw = float(state_row["line_capacity_mw"]) if state_row is not None else float(data["capacity_mw"])
        flow_mw = float(state_row["line_flow_mw"]) if state_row is not None else float(data["current_flow_mw"])
        loading_percent = 0.0 if capacity_mw <= 0 else (flow_mw / capacity_mw) * 100.0
        metadata["measured_line_loading"][line_id] = {
            "line_id": line_id,
            "edge_type": data.get("edge_type", "transmission_line"),
            "capacity_mw": capacity_mw,
            "current_flow_mw": flow_mw,
            "loading_percent": round(loading_percent, 2),
            "from_bus": from_bus,
            "to_bus": to_bus,
        }

        ampacity_capacity_mw = capacity_mw * 3.0 if data.get("edge_type") == "transformer" else capacity_mw
        max_i_ka = _capacity_to_max_i_ka(ampacity_capacity_mw, DEFAULT_VN_KV)
        resistance = max(float(data["impedance"]) * 0.35, 0.01)
        reactance = max(float(data["impedance"]) * 0.65, 0.01)
        line_index = pp.create_line_from_parameters(
            net,
            from_bus=metadata["bus_map"][from_bus],
            to_bus=metadata["bus_map"][to_bus],
            length_km=0.2 if data.get("edge_type") == "transformer" else 1.0,
            r_ohm_per_km=resistance,
            x_ohm_per_km=reactance,
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=line_id,
            in_service=state_row is None or str(state_row["fault_status"]) != "fault",
        )
        net.line.loc[line_index, "line_id"] = line_id
        net.line.loc[line_index, "edge_type"] = data.get("edge_type", "transmission_line")
        net.line.loc[line_index, "max_loading_percent"] = 100.0
        metadata["line_map"][line_id] = int(line_index)

    _attach_default_costs(net, metadata)
    return net, metadata


def _attach_default_costs(net: Any, metadata: dict[str, Any]) -> None:
    _require_pandapower()
    pp.create_poly_cost(net, metadata["ext_grid_index"], "ext_grid", cp1_eur_per_mw=20.0)

    for generator_index in metadata["generator_map"].values():
        pp.create_poly_cost(net, generator_index, "gen", cp1_eur_per_mw=12.0)

    for renewable_index in metadata["renewable_map"].values():
        pp.create_poly_cost(net, renewable_index, "sgen", cp1_eur_per_mw=2.0)

    for storage_index in metadata["storage_map"].values():
        pp.create_poly_cost(net, storage_index, "storage", cp1_eur_per_mw=6.0)


def _capacity_to_max_i_ka(capacity_mw: float, vn_kv: float) -> float:
    apparent_power_mva = max(capacity_mw, 0.1)
    return max(apparent_power_mva / (math.sqrt(3.0) * vn_kv), 0.05)
