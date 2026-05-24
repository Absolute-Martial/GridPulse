"""Pandapower-backed power-flow and OPF validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.digital_twin.grid_graph import build_default_grid, get_grid_summary, get_overloaded_lines, update_grid_state
from app.optimization.pandapower_adapter import build_pandapower_network

try:
    import pandapower as pp
    from pandapower.auxiliary import LoadflowNotConverged, OPFNotConverged
except ModuleNotFoundError:
    pp = None
    LoadflowNotConverged = RuntimeError  # type: ignore[assignment]
    OPFNotConverged = RuntimeError  # type: ignore[assignment]


def _require_pandapower() -> None:
    if pp is None:
        raise ImportError("pandapower is required for GridPulse optimization validation but is not installed.")


@dataclass
class PandapowerOPFSolver:
    """Run pandapower validation on top of the GridPulse digital twin."""

    last_report: dict[str, Any] | None = field(default=None, init=False)

    def build_pandapower_network(
        self,
        grid_graph=None,
        sensor_dataframe: pd.DataFrame | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        _require_pandapower()
        if sensor_dataframe is None or sensor_dataframe.empty:
            raise ValueError("Sensor dataframe is empty.")
        graph = grid_graph or build_default_grid()
        update_grid_state(sensor_dataframe)
        return build_pandapower_network(graph, sensor_dataframe)

    def run_power_flow(
        self,
        grid_graph=None,
        sensor_dataframe: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        _require_pandapower()
        net, metadata = self.build_pandapower_network(grid_graph=grid_graph, sensor_dataframe=sensor_dataframe)

        try:
            pp.runpp(net, init="flat", calculate_voltage_angles=False)
            report = self.return_feasibility_report(net, metadata, mode="power_flow")
        except LoadflowNotConverged as exc:
            report = self._failed_report(net, metadata, "power_flow", f"Power flow did not converge: {exc}")

        self.last_report = report
        return report

    def run_opf(
        self,
        grid_graph=None,
        sensor_dataframe: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        _require_pandapower()
        net, metadata = self.build_pandapower_network(grid_graph=grid_graph, sensor_dataframe=sensor_dataframe)

        try:
            pp.runopp(net, verbose=False)
            report = self.return_feasibility_report(net, metadata, mode="opf")
        except OPFNotConverged as exc:
            report = self._failed_report(net, metadata, "opf", f"OPF did not converge: {exc}")
        except Exception as exc:
            report = self._failed_report(net, metadata, "opf", f"OPF failed: {exc}")

        self.last_report = report
        return report

    def validate_dispatch(
        self,
        sensor_dataframe: pd.DataFrame,
        dispatch_plan: dict[str, Any] | None = None,
        grid_graph=None,
    ) -> dict[str, Any]:
        _require_pandapower()
        net, metadata = self.build_pandapower_network(grid_graph=grid_graph, sensor_dataframe=sensor_dataframe)
        dispatch_plan = dispatch_plan or {}
        self._apply_dispatch_plan(net, metadata, dispatch_plan)

        if self._dispatch_is_obviously_infeasible(net):
            report = self._failed_report(
                net,
                metadata,
                "validate_dispatch",
                "Dispatch plan cannot supply the requested load within configured generator and ext_grid limits.",
            )
            self.last_report = report
            return report

        try:
            pp.runopp(net, verbose=False)
            report = self.return_feasibility_report(net, metadata, mode="validate_dispatch")
        except OPFNotConverged as exc:
            report = self._failed_report(net, metadata, "validate_dispatch", f"Dispatch validation OPF did not converge: {exc}")
        except Exception as exc:
            report = self._failed_report(net, metadata, "validate_dispatch", f"Dispatch validation failed: {exc}")

        self.last_report = report
        return report

    def calculate_line_loading(self, net: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        measured_map = metadata.get("measured_line_loading", {})
        line_results: list[dict[str, Any]] = []
        res_line = getattr(net, "res_line", None)

        for line_id, line_index in metadata["line_map"].items():
            measured = measured_map.get(line_id, {})
            loading_percent = float(measured.get("loading_percent", 0.0))
            current_flow_mw = float(measured.get("current_flow_mw", 0.0))
            capacity_mw = float(measured.get("capacity_mw", 0.0))
            losses_mw = 0.0

            if res_line is not None and not res_line.empty and line_index in res_line.index:
                result_row = res_line.loc[line_index]
                loading_percent = max(loading_percent, float(result_row.get("loading_percent", 0.0)))
                losses_mw = float(result_row.get("pl_mw", 0.0))

            line_results.append(
                {
                    "line_id": line_id,
                    "edge_type": measured.get("edge_type", str(net.line.loc[line_index].get("edge_type", "transmission_line"))),
                    "from_bus": measured.get("from_bus"),
                    "to_bus": measured.get("to_bus"),
                    "capacity_mw": capacity_mw,
                    "current_flow_mw": current_flow_mw,
                    "loading_percent": round(loading_percent, 2),
                    "losses_mw": round(losses_mw, 4),
                    "is_overloaded": loading_percent > 100.0,
                }
            )

        return sorted(line_results, key=lambda item: item["loading_percent"], reverse=True)

    def calculate_voltage_violations(self, net: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        measured_voltage = metadata.get("measured_bus_voltage", {})
        violations: list[dict[str, Any]] = []

        for bus_name, bus_index in metadata["bus_map"].items():
            bus_row = net.bus.loc[bus_index]
            vm_min = float(bus_row.get("min_vm_pu", 0.95))
            vm_max = float(bus_row.get("max_vm_pu", 1.05))
            actual_vm = None

            if hasattr(net, "res_bus") and not net.res_bus.empty and bus_index in net.res_bus.index:
                actual_vm = float(net.res_bus.loc[bus_index].get("vm_pu", float("nan")))
                if pd.notna(actual_vm) and (actual_vm < vm_min or actual_vm > vm_max):
                    violations.append(
                        {
                            "bus_id": bus_name,
                            "vm_pu": round(actual_vm, 4),
                            "min_vm_pu": vm_min,
                            "max_vm_pu": vm_max,
                            "source": "pandapower",
                        }
                    )

            measured_vm = float(measured_voltage.get(bus_name, 1.0))
            if measured_vm < vm_min or measured_vm > vm_max:
                if not any(item["bus_id"] == bus_name for item in violations):
                    violations.append(
                        {
                            "bus_id": bus_name,
                            "vm_pu": round(measured_vm, 4),
                            "min_vm_pu": vm_min,
                            "max_vm_pu": vm_max,
                            "source": "sensor_snapshot",
                        }
                    )

        return violations

    def calculate_unsupplied_load(self, net: Any, metadata: dict[str, Any]) -> float:
        requested = sum(float(value) for value in metadata.get("requested_load_mw", {}).values())
        if not hasattr(net, "res_load") or net.res_load.empty:
            return round(requested, 4)
        served = float(net.res_load["p_mw"].sum())
        return round(max(requested - served, 0.0), 4)

    def return_feasibility_report(
        self,
        net: Any,
        metadata: dict[str, Any],
        mode: str,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        line_loading = self.calculate_line_loading(net, metadata)
        overloaded_lines = [item for item in line_loading if item["is_overloaded"]]
        voltage_violations = self.calculate_voltage_violations(net, metadata)
        generation_limit_violations = self._calculate_generation_limit_violations(net)
        unsupplied_load_mw = self.calculate_unsupplied_load(net, metadata)
        total_losses_mw = round(float(net.res_line["pl_mw"].sum()) if hasattr(net, "res_line") and not net.res_line.empty else 0.0, 4)
        is_feasible = (
            failure_reason is None
            and not overloaded_lines
            and not voltage_violations
            and not generation_limit_violations
            and unsupplied_load_mw <= 1e-3
        )

        return {
            "mode": mode,
            "is_feasible": is_feasible,
            "failure_reason": failure_reason,
            "voltage_violations": voltage_violations,
            "overloaded_lines": overloaded_lines,
            "generation_limit_violations": generation_limit_violations,
            "unsupplied_load_mw": unsupplied_load_mw,
            "total_losses_mw": total_losses_mw,
            "line_loading": line_loading,
            "grid_summary": get_grid_summary(),
            "recommended_corrective_action": self._recommended_corrective_action(
                failure_reason=failure_reason,
                overloaded_lines=overloaded_lines,
                voltage_violations=voltage_violations,
                unsupplied_load_mw=unsupplied_load_mw,
                generation_limit_violations=generation_limit_violations,
            ),
        }

    def _apply_dispatch_plan(self, net: Any, metadata: dict[str, Any], dispatch_plan: dict[str, Any]) -> None:
        ext_grid_plan = dispatch_plan.get("ext_grid", {})
        ext_grid_index = metadata.get("ext_grid_index")
        if ext_grid_index is not None:
            for key, value in ext_grid_plan.items():
                if key in net.ext_grid.columns:
                    net.ext_grid.loc[ext_grid_index, key] = value

        for generator_id, config in dispatch_plan.get("generators", {}).items():
            if generator_id not in metadata["generator_map"]:
                continue
            generator_index = metadata["generator_map"][generator_id]
            for key, value in config.items():
                if key in net.gen.columns:
                    net.gen.loc[generator_index, key] = value

        for renewable_id, config in dispatch_plan.get("renewables", {}).items():
            if renewable_id not in metadata["renewable_map"]:
                continue
            renewable_index = metadata["renewable_map"][renewable_id]
            for key, value in config.items():
                if key in net.sgen.columns:
                    net.sgen.loc[renewable_index, key] = value

        for zone_id, config in dispatch_plan.get("loads", {}).items():
            if zone_id not in metadata["load_map"]:
                continue
            load_index = metadata["load_map"][zone_id]
            for key, value in config.items():
                if key in net.load.columns:
                    net.load.loc[load_index, key] = value

        for storage_id, config in dispatch_plan.get("storage", {}).items():
            if storage_id not in metadata["storage_map"]:
                continue
            storage_index = metadata["storage_map"][storage_id]
            for key, value in config.items():
                if key in net.storage.columns:
                    net.storage.loc[storage_index, key] = value

    def _dispatch_is_obviously_infeasible(self, net: Any) -> bool:
        total_load = float(net.load["p_mw"].sum()) if not net.load.empty else 0.0
        total_supply_limit = 0.0
        if not net.ext_grid.empty:
            total_supply_limit += float(net.ext_grid["max_p_mw"].fillna(0.0).sum())
        if not net.gen.empty:
            total_supply_limit += float(net.gen["max_p_mw"].fillna(0.0).sum())
        if not net.sgen.empty:
            total_supply_limit += float(net.sgen["max_p_mw"].fillna(0.0).sum())
        if not net.storage.empty:
            discharge_capability = net.storage["min_p_mw"].fillna(0.0).abs().sum()
            total_supply_limit += float(discharge_capability)
        return total_supply_limit + 1e-6 < total_load

    def _calculate_generation_limit_violations(self, net: Any) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        violations.extend(self._element_limit_violations(net, "gen", "res_gen", "p_mw"))
        violations.extend(self._element_limit_violations(net, "sgen", "res_sgen", "p_mw"))
        violations.extend(self._element_limit_violations(net, "ext_grid", "res_ext_grid", "p_mw"))
        violations.extend(self._element_limit_violations(net, "storage", "res_storage", "p_mw"))
        return violations

    def _element_limit_violations(self, net: Any, table_name: str, result_name: str, power_column: str) -> list[dict[str, Any]]:
        if not hasattr(net, result_name) or getattr(net, result_name).empty:
            return []
        table = getattr(net, table_name)
        result = getattr(net, result_name)
        violations: list[dict[str, Any]] = []
        for index, row in result.iterrows():
            if index not in table.index:
                continue
            min_col = f"min_{power_column}"
            max_col = f"max_{power_column}"
            min_value = float(table.loc[index, min_col]) if min_col in table.columns and pd.notna(table.loc[index, min_col]) else None
            max_value = float(table.loc[index, max_col]) if max_col in table.columns and pd.notna(table.loc[index, max_col]) else None
            actual = float(row.get(power_column, 0.0))

            if min_value is not None and actual < min_value - 1e-4:
                violations.append({"element_type": table_name, "element": str(table.loc[index, "name"]), "actual_p_mw": actual, "limit": min_value, "violation": "below_min"})
            if max_value is not None and actual > max_value + 1e-4:
                violations.append({"element_type": table_name, "element": str(table.loc[index, "name"]), "actual_p_mw": actual, "limit": max_value, "violation": "above_max"})
        return violations

    def _recommended_corrective_action(
        self,
        failure_reason: str | None,
        overloaded_lines: list[dict[str, Any]],
        voltage_violations: list[dict[str, Any]],
        unsupplied_load_mw: float,
        generation_limit_violations: list[dict[str, Any]],
    ) -> str:
        if failure_reason:
            return "Relax dispatch limits, increase slack support, or reduce controllable load before rerunning OPF."
        if overloaded_lines:
            return "Reroute power or reduce load on overloaded corridors before dispatch execution."
        if voltage_violations:
            return "Adjust generator voltage targets or provide reactive support near violated buses."
        if unsupplied_load_mw > 0.0:
            return "Increase dispatchable generation or battery discharge to cover unsupplied load."
        if generation_limit_violations:
            return "Rebalance generator setpoints to keep dispatch within operational limits."
        return "No corrective action required."

    def _failed_report(self, net: Any, metadata: dict[str, Any], mode: str, reason: str) -> dict[str, Any]:
        report = self.return_feasibility_report(net, metadata, mode=mode, failure_reason=reason)
        report["is_feasible"] = False
        return report


LAST_FEASIBILITY_REPORT: dict[str, Any] | None = None

