"""Abstract forecasting interfaces for GridPulse."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseForecaster(ABC):
    """Common interface for swappable offline forecasters."""

    model_name: str

    @abstractmethod
    def train(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        """Train the model for one forecast target and horizon."""

    @abstractmethod
    def predict(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        """Return horizon-step predictions for one forecast target."""

    @abstractmethod
    def evaluate(
        self,
        dataframe: pd.DataFrame,
        entity_type: str,
        entity_id: str,
        horizon: str,
    ) -> dict[str, Any]:
        """Evaluate the model on held-out local data."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained artifact."""

    @abstractmethod
    def load(self, path: Path) -> "BaseForecaster":
        """Load the trained artifact."""
