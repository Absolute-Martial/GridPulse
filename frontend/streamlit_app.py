"""Self-contained Streamlit Cloud app for GridPulse."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:  # pragma: no cover - optional live-refresh dependency
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - fallback without auto-refresh
    st_autorefresh = None

from streamlit_runtime import (
    LoadedArtifact,
    append_live_step,
    available_model_paths,
    build_explanation,
    build_streamlit_forecast,
    build_telemetry_snapshot,
    format_artifact_label,
    format_metric_rows,
    load_artifact,
    load_initial_history,
)


st.set_page_config(
    page_title="GridPulse Forecast Studio",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_artifact(path: str | None = None) -> LoadedArtifact:
    return load_artifact(Path(path) if path else None)


@st.cache_data(show_spinner=False)
def cached_history(artifact_path: str) -> pd.DataFrame:
    artifact = load_artifact(Path(artifact_path))
    return load_initial_history(artifact)


def main() -> None:
    st.title("GridPulse Forecast Studio")
    st.caption(
        "Self-contained Streamlit Cloud demo that loads the packaged `.pkl` model, simulates a live 15-minute AMI stream, and renders p10/p90 bands."
    )

    model_paths = available_model_paths()
    if not model_paths:
        st.error("No forecast artifact found in data/models/forecasting or kaggle/output.")
        st.stop()

    default_path = st.session_state.get("active_model_path", str(model_paths[0]))
    if default_path not in {str(path) for path in model_paths}:
        default_path = str(model_paths[0])

    with st.sidebar:
        st.header("Controls")
        selected_path = st.selectbox(
            "Model artifact",
            options=[str(path) for path in model_paths],
            index=[str(path) for path in model_paths].index(default_path),
            format_func=lambda value: format_artifact_label(Path(value), cached_artifact(value)),
        )
        auto_advance = st.checkbox("Auto-advance live stream", value=True)
        refresh_seconds = st.slider("Refresh seconds", min_value=5, max_value=60, value=15, step=5)
        advance_now = st.button("Advance 15 minutes")
        reset_stream = st.button("Reset stream")
        st.divider()
        artifact = cached_artifact(selected_path)
        st.metric("Selected model", artifact.model_name)
        st.metric("Target", f"{artifact.entity_type} / {artifact.entity_id}")
        st.metric("Horizon", artifact.horizon)
        st.metric("Model file", Path(selected_path).name)

    if st.session_state.get("active_model_path") != selected_path or "history" not in st.session_state:
        st.session_state["active_model_path"] = selected_path
        st.session_state["history"] = cached_history(selected_path).copy()
        st.session_state["initialized"] = False

    artifact = cached_artifact(selected_path)
    history = st.session_state["history"].copy()

    if reset_stream:
        st.session_state["history"] = cached_history(selected_path).copy()
        st.session_state["initialized"] = False
        st.rerun()

    if auto_advance and st.session_state.get("initialized", False):
        if st_autorefresh is not None:
            st_autorefresh(interval=refresh_seconds * 1000, key="gridpulse-auto-refresh")
        history = append_live_step(history)
        st.session_state["history"] = history
    elif advance_now:
        history = append_live_step(history)
        st.session_state["history"] = history

    st.session_state["initialized"] = True

    forecast = build_streamlit_forecast(history, artifact)
    telemetry = build_telemetry_snapshot(history)
    explanation = build_explanation(history, forecast)
    model_metrics = artifact.metrics.get("metrics", artifact.metrics)

    latest_row = history.sort_values("timestamp").iloc[-1]
    latest_timestamp = pd.to_datetime(latest_row["timestamp"], utc=True).strftime("%Y-%m-%d %H:%M")

    st.subheader("Live telemetry")
    telemetry_columns = st.columns(6)
    telemetry_values = [
        ("Latest load", telemetry["load_kw"], "kW"),
        ("Voltage", telemetry["voltage_pu"], "pu"),
        ("Frequency", telemetry["frequency_hz"], "Hz"),
        ("Temperature", telemetry["temperature_c"], "C"),
        ("Humidity", telemetry["humidity_percent"], "%"),
        ("Storage SOC", telemetry["storage_soc_percent"], "%"),
    ]
    for column, (label, value, unit) in zip(telemetry_columns, telemetry_values, strict=True):
        display_value = f"{value} {unit}".strip() if unit else str(value)
        column.metric(label, display_value)

    st.subheader("Live generated AMI stream")
    live_frame = history.tail(12).copy()
    st.dataframe(
        live_frame.loc[
            :,
            [
                "timestamp",
                "entity_type",
                "entity_id",
                "load_kw",
                "fingerprint_mean_kw",
                "temperature_c",
                "humidity_percent",
                "source_type",
                "data_quality_flag",
            ],
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Forecast")
    forecast_columns = st.columns(4)
    forecast_columns[0].metric("Peak load", f"{forecast['summary']['peak_load_kw']} kW")
    forecast_columns[1].metric("Peak time", forecast["summary"]["peak_time"])
    forecast_columns[2].metric("Mean load", f"{forecast['summary']['mean_load_kw']} kW")
    forecast_columns[3].metric("Confidence", forecast["summary"]["confidence"])

    st.caption(
        f"Latest observed timestamp: {latest_timestamp} | lookback: {forecast['lookback_steps']} steps | horizon: {forecast['horizon']} ({forecast['horizon_steps']} slots)"
    )

    st.plotly_chart(render_forecast_chart(history, forecast), use_container_width=True)
    st.plotly_chart(render_actual_vs_predicted_chart(history, forecast), use_container_width=True)

    st.subheader("Slot predictions")
    st.dataframe(pd.DataFrame(forecast["slot_predictions"]), use_container_width=True, hide_index=True)

    st.subheader("Aggregated summary")
    st.dataframe(pd.DataFrame(forecast["aggregated_summary"]), use_container_width=True, hide_index=True)

    st.subheader("Plain-language explanation")
    st.write(explanation)

    st.subheader("Artifact metrics")
    st.dataframe(pd.DataFrame(format_metric_rows(artifact.metrics)), use_container_width=True, hide_index=True)

    with st.expander("Research reference", expanded=False):
        st.markdown(
            """
GridPulse forecasting is grounded in the local research docs that shaped the current MVP:

- `GridPulse_Modular_Prototype_Implementation_Paper.docx`
- `GridPulse_Technical_Architecture.docx`

The Streamlit demo uses those documents as the source reference for the offline,
15-minute AMI forecasting flow and the p10/p90 visualization behavior.
            """
        )

    with st.expander("Artifact details", expanded=False):
        st.write(f"Artifact: `{artifact.path}`")
        st.write(f"Feature columns: {len(artifact.feature_columns)}")
        st.write(f"Latest input timestamp: {forecast['latest_input_timestamp']}")
        st.write(f"Source entity: {artifact.entity_type} / {artifact.entity_id}")
        st.write(f"Model version: {artifact.model_name}")
        if model_metrics:
            st.json(model_metrics)


def render_forecast_chart(history: pd.DataFrame, forecast: dict) -> go.Figure:
    sorted_history = history.sort_values("timestamp").copy()
    recent_history = sorted_history.tail(96)
    recent_history["timestamp"] = pd.to_datetime(recent_history["timestamp"], utc=True)

    slot_frame = pd.DataFrame(forecast["slot_predictions"]).copy()
    slot_frame["timestamp"] = pd.to_datetime(slot_frame["timestamp"], utc=True)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recent_history["timestamp"],
            y=recent_history["load_kw"],
            mode="lines",
            name="Observed load",
            line=dict(color="#1f2937", width=2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=slot_frame["timestamp"],
            y=slot_frame["p90_kw"],
            mode="lines",
            line=dict(width=0),
            name="p90",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=slot_frame["timestamp"],
            y=slot_frame["p10_kw"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line=dict(width=0),
            name="p10-p90 band",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=slot_frame["timestamp"],
            y=slot_frame["predicted_load_kw"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=slot_frame["timestamp"],
            y=slot_frame["fingerprint_mean_kw"],
            mode="lines+markers",
            name="Fingerprint baseline",
            line=dict(color="#d97706", width=2, dash="dot"),
            marker=dict(size=5),
        )
    )
    figure.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Timestamp",
        yaxis_title="Load (kW)",
    )
    return figure


def render_actual_vs_predicted_chart(history: pd.DataFrame, forecast: dict) -> go.Figure:
    """Render actual live generation versus forecast and the forecast gap."""

    observed = history.sort_values("timestamp").tail(96).copy()
    observed["timestamp"] = pd.to_datetime(observed["timestamp"], utc=True)

    slot_frame = pd.DataFrame(forecast["slot_predictions"]).copy()
    slot_frame["timestamp"] = pd.to_datetime(slot_frame["timestamp"], utc=True)
    slot_frame["forecast_gap_kw"] = slot_frame["predicted_load_kw"] - slot_frame["fingerprint_mean_kw"]

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=observed["timestamp"],
            y=observed["load_kw"],
            mode="lines",
            name="Actual generated load",
            line=dict(color="#111827", width=2),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=slot_frame["timestamp"],
            y=slot_frame["predicted_load_kw"],
            mode="lines+markers",
            name="Predicted load",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=slot_frame["timestamp"],
            y=slot_frame["fingerprint_mean_kw"],
            mode="lines+markers",
            name="Fingerprint mean",
            line=dict(color="#d97706", width=2, dash="dot"),
            marker=dict(size=5),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=slot_frame["timestamp"],
            y=slot_frame["forecast_gap_kw"],
            name="Predicted - fingerprint mean",
            marker_color="rgba(220, 38, 38, 0.35)",
        ),
        secondary_y=True,
    )
    figure.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="rgba(127, 29, 29, 0.6)", secondary_y=True)
    figure.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Timestamp",
        yaxis_title="Load (kW)",
        yaxis2_title="Forecast gap (kW)",
    )
    return figure


if __name__ == "__main__":
    main()
