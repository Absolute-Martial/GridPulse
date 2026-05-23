"""GridPulse Streamlit dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard.components.network_plot import network_figure
from dashboard.components.shap_panel import render_shap_waterfall
from dashboard.components.suggestion_cards import render_suggestions
from gridpulse.config import load_settings
from gridpulse.engine import GridPulseEngine

settings = load_settings()
st.set_page_config(page_title=settings.app.name, page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner="Booting GridPulse…")
def _bootstrap() -> GridPulseEngine:
    return GridPulseEngine(settings=settings)


def _init_state() -> None:
    if "engine" in st.session_state:
        return
    st.session_state.engine = _bootstrap()
    st.session_state.history = []
    st.session_state.last_snapshot = None


_init_state()
engine: GridPulseEngine = st.session_state.engine

st.title("GridPulse — Explainable AI Co-Pilot for Smart Grid Load Balancing")
st.caption("NetworkX-first hackathon prototype with optional pandapower validation and OpenEMS as the baseline reference.")

with st.expander("How this prototype is scoped", expanded=False):
    st.markdown(
        """
        GridPulse is a decision-support prototype, not a live SCADA controller.
        The default path is deliberately lightweight:

        - NetworkX graph for routing and dispatch
        - Random-forest short-horizon forecasting
        - LightGBM grid-state classification on UCI-style features
        - Isolation Forest anomaly scoring
        - SHAP explanations for operator guidance
        - Optional pandapower validation left off by default
        """
    )
    st.json(engine.grid_summary())

left, middle, right = st.columns([1.4, 1.2, 2.0])

with left:
    st.subheader("Simulation")
    if st.button("Advance one tick", use_container_width=True, type="primary"):
        snapshot = engine.advance_tick()
        st.session_state.last_snapshot = snapshot
        st.session_state.history.append(
            {
                "tick": snapshot.tick,
                "state": snapshot.prediction.label,
                "anomaly": snapshot.anomaly_score,
                "demand_kw": snapshot.dispatch.total_demand_kw,
                "supplied_kw": snapshot.dispatch.total_supplied_kw,
                "renewable_share": snapshot.dispatch.renewable_share,
                "max_utilization": snapshot.dispatch.max_line_utilization,
            }
        )

    scenario = st.selectbox("Scenario", settings.scenario.available, index=settings.scenario.available.index(engine.simulator.scenario_name))
    if st.button("Apply scenario", use_container_width=True):
        engine.set_scenario(scenario)

with middle:
    st.subheader("Operator controls")
    fault_target = st.selectbox("Fault target", [node for node, attrs in engine.graph.nodes(data=True) if attrs.get("kind") == "substation"])
    control_a, control_b = st.columns(2)
    if control_a.button("Inject fault", use_container_width=True):
        engine.inject_fault(fault_target)
    if control_b.button("Clear fault", use_container_width=True):
        engine.clear_fault(fault_target)

    if st.button("Jump to renewable surplus", use_container_width=True):
        engine.set_scenario("renewable_surplus")
    if st.button("Jump to peak stress", use_container_width=True):
        engine.set_scenario("peak")

with right:
    st.subheader("Live KPIs")
    snapshot = st.session_state.last_snapshot
    if snapshot is None:
        st.info("Advance one tick to generate the first operator snapshot.")
    else:
        top = st.columns(4)
        top[0].metric("Grid state", snapshot.prediction.label, f"{int(snapshot.prediction.probabilities.get(snapshot.prediction.label, 0)*100)}%")
        top[1].metric("Anomaly", f"{snapshot.anomaly_score:.2f}")
        top[2].metric("Demand", f"{snapshot.dispatch.total_demand_kw:.0f} kW")
        top[3].metric("Supplied", f"{snapshot.dispatch.total_supplied_kw:.0f} kW")
        bottom = st.columns(4)
        bottom[0].metric("Renewables", f"{snapshot.dispatch.renewable_share*100:.0f}%")
        bottom[1].metric("Max utilization", f"{snapshot.dispatch.max_line_utilization:.0f}%")
        bottom[2].metric("Faults", len(snapshot.faulted_nodes))
        bottom[3].metric("Scenario", snapshot.scenario.name)

st.divider()

main_left, main_right = st.columns([1.55, 1.0])

with main_left:
    st.subheader("Grid topology")
    st.plotly_chart(network_figure(engine.graph, faulted_nodes=st.session_state.last_snapshot.faulted_nodes if st.session_state.last_snapshot else []), use_container_width=True)

    if st.session_state.history:
        history = pd.DataFrame(st.session_state.history).tail(48).set_index("tick")
        st.subheader("History")
        st.line_chart(history[["demand_kw", "supplied_kw", "max_utilization"]])
        st.line_chart(history[["anomaly", "renewable_share"]])

with main_right:
    st.subheader("Top operator suggestions")
    render_suggestions(st.session_state.last_snapshot.suggestions if st.session_state.last_snapshot else [])

    if st.session_state.last_snapshot:
        with st.expander("SHAP explanation", expanded=False):
            latest_features = engine.simulator._uci_like_features(  # noqa: SLF001
                st.session_state.last_snapshot.zone_states,
                st.session_state.last_snapshot.dispatch.total_demand_kw,
                engine.profiles["solar_factor"][max(engine.simulator.tick - 1, 0) % 24],
                engine.profiles["wind_factor"][max(engine.simulator.tick - 1, 0) % 24],
            )
            render_shap_waterfall(engine.classifier, latest_features, st.session_state.last_snapshot.prediction.label)

with st.expander("Dispatch diagnostics", expanded=False):
    if st.session_state.last_snapshot:
        st.write("Line flows")
        st.dataframe(pd.DataFrame(st.session_state.last_snapshot.dispatch.line_flows))
        st.write("Routes")
        st.json(st.session_state.last_snapshot.dispatch.routes)
        if st.session_state.last_snapshot.dispatch.validation:
            st.write("Pandapower validation")
            st.json(st.session_state.last_snapshot.dispatch.validation)
