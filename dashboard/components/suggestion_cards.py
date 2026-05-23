"""Suggestion card renderer for Streamlit."""
from __future__ import annotations

from typing import List

import streamlit as st

from gridpulse.schemas import Suggestion

_PRIORITY_STYLE = {
    "HIGH":   ("#ffebee", "#c62828", "🔴"),
    "MEDIUM": ("#fff8e1", "#f9a825", "🟡"),
    "LOW":    ("#e3f2fd", "#1565c0", "🔵"),
}


def render_suggestions(suggestions: List[Suggestion]) -> None:
    if not suggestions:
        st.info("No suggestions for this tick.")
        return
    for s in suggestions:
        bg, fg, icon = _PRIORITY_STYLE.get(s.priority, ("#f5f5f5", "#424242", "•"))
        with st.container():
            st.markdown(
                f"""
                <div style="background:{bg};border-left:6px solid {fg};
                            padding:12px 16px;border-radius:6px;margin-bottom:10px;">
                    <div style="font-size:13px;color:{fg};font-weight:700;letter-spacing:0.05em;">
                        {icon} {s.priority} · {int(s.confidence*100)}% confidence
                    </div>
                    <div style="font-size:17px;font-weight:600;margin:4px 0 6px 0;">
                        {s.title}
                    </div>
                    <div style="font-size:14px;color:#444;margin-bottom:4px;">
                        <b>Why:</b> {s.why}
                    </div>
                    <div style="font-size:13px;color:#666;">
                        <b>Estimated impact:</b> {s.est_impact}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if s.top_features:
                feats = " · ".join(
                    f"<code>{name}={val:+.2f}</code>" for name, val in s.top_features[:3]
                )
                st.markdown(
                    f"<div style='font-size:12px;color:#777;margin:-6px 0 14px 6px;'>SHAP drivers: {feats}</div>",
                    unsafe_allow_html=True,
                )
