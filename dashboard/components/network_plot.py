"""Plotly topology rendering for the NetworkX-first dashboard."""
from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go


NODE_STYLE = {
    "generator": ("#1b5e20", 22),
    "battery": ("#ef6c00", 20),
    "substation": ("#1565c0", 18),
    "zone": ("#6a1b9a", 16),
}


def network_figure(graph: nx.Graph, faulted_nodes: list[str] | None = None, title: str = "GridPulse Network") -> go.Figure:
    faulted_nodes = faulted_nodes or []
    positions = nx.spring_layout(graph, seed=9, weight="resistance")
    fig = go.Figure()

    for left, right, attrs in graph.edges(data=True):
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=2, color="#b0bec5"),
                hovertemplate=f"{left} ↔ {right}<br>capacity={attrs['capacity_kw']:.0f} kW<br>resistance={attrs['resistance']:.2f}<extra></extra>",
                showlegend=False,
            )
        )

    by_kind: dict[str, list[str]] = {}
    for node_id, attrs in graph.nodes(data=True):
        by_kind.setdefault(attrs.get("kind", "zone"), []).append(node_id)

    for kind, nodes in by_kind.items():
        color, size = NODE_STYLE.get(kind, ("#455a64", 16))
        xs, ys, text = [], [], []
        for node_id in nodes:
            x, y = positions[node_id]
            xs.append(x)
            ys.append(y)
            label = node_id
            if node_id in faulted_nodes:
                label = f"{node_id} (fault)"
            text.append(label)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=text,
                textposition="top center",
                marker=dict(
                    size=[size + 6 if node in faulted_nodes else size for node in nodes],
                    color=["#d32f2f" if node in faulted_nodes else color for node in nodes],
                    line=dict(width=2, color="#ffffff"),
                ),
                name=kind.title(),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
