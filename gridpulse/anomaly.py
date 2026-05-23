"""Anomaly detection — drop-in sklearn IsolationForest on UCI features.

Identical-shape feature vector as the classifier. We refit periodically on
the rolling window of recent ticks. The score is a calibrated [0,1] number
where higher means more anomalous, exposed to the suggestion engine.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from gridpulse.data_loader import feature_columns


class AnomalyDetector:
    def __init__(self, window: int = 200, contamination: float = 0.05, refit_every: int = 50):
        self.window = window
        self.contamination = contamination
        self.refit_every = refit_every
        self.history: Deque[np.ndarray] = deque(maxlen=window)
        self.model: Optional[IsolationForest] = None
        self._since_refit = 0

    def update(self, features: pd.Series | np.ndarray) -> float:
        if isinstance(features, pd.Series):
            x = features[feature_columns()].values.astype(float)
        else:
            x = np.asarray(features, dtype=float)
        self.history.append(x)
        self._since_refit += 1

        if len(self.history) < 30:
            return 0.0

        if self.model is None or self._since_refit >= self.refit_every:
            X = np.vstack(list(self.history))
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=120,
            )
            self.model.fit(X)
            self._since_refit = 0

        raw = self.model.decision_function(x.reshape(1, -1))[0]
        # Map raw decision score (~ [-0.5, 0.5]) to anomaly score in [0, 1]
        return float(np.clip(0.5 - raw, 0.0, 1.0))
