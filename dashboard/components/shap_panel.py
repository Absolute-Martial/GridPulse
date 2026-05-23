"""SHAP waterfall renderer for Streamlit (PNG via matplotlib)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from gridpulse.classifier import LABELS, GridStateClassifier
from gridpulse.data_loader import feature_columns


def render_shap_waterfall(classifier: GridStateClassifier, features, predicted_label: str) -> None:
    import shap

    cols = feature_columns()
    x = features[cols].values.reshape(1, -1)
    explainer = shap.TreeExplainer(classifier.model)
    shap_values = explainer.shap_values(x)

    cls_idx = LABELS.index(predicted_label) if predicted_label in LABELS else 0
    if isinstance(shap_values, list):
        sv = shap_values[cls_idx][0]
        base = explainer.expected_value[cls_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        sv_arr = np.asarray(shap_values)
        if sv_arr.ndim == 3:
            sv = sv_arr[0, :, cls_idx]
            base = explainer.expected_value[cls_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            sv = sv_arr[0]
            base = float(np.atleast_1d(explainer.expected_value)[0])

    explanation = shap.Explanation(
        values=sv,
        base_values=float(np.atleast_1d(base)[0]) if hasattr(base, '__len__') else float(base),
        data=x[0],
        feature_names=cols,
    )

    plt.close("all")
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(explanation, max_display=8, show=False)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, caption=f"SHAP waterfall — class: {predicted_label}", use_column_width=True)
    plt.close(fig)
