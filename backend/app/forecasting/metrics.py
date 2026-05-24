"""Regression metrics for offline forecasting."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = np.where(actual_array == 0, 1e-6, actual_array)

    peak_time_error = float(abs(int(actual_array.argmax()) - int(predicted_array.argmax())))
    peak_load_error = float(abs(actual_array.max() - predicted_array.max()))

    return {
        "mae": round(float(mean_absolute_error(actual_array, predicted_array)), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(actual_array, predicted_array))), 4),
        "mape": round(
            float(np.mean(np.abs((actual_array - predicted_array) / denominator)) * 100.0),
            4,
        ),
        "r2": round(float(r2_score(actual_array, predicted_array)), 4),
        "peak_time_error": round(peak_time_error, 4),
        "peak_load_error": round(peak_load_error, 4),
    }
