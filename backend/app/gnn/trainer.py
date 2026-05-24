"""Training helpers for the GridPulse topology-risk GNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.core.config import get_settings
from app.gnn.graph_dataset import EDGE_FEATURE_COLUMNS, NODE_FEATURE_COLUMNS, SyntheticGraphScenarioDataset, generate_scenario_dataset
from app.gnn.models import GridRiskGNN


@dataclass
class GridRiskGNNTrainer:
    """Train and persist the topology-risk GNN artifact."""

    model_path: Path | None = None

    def __post_init__(self) -> None:
        self.model_path = self.model_path or get_settings().gnn_model_path

    def train(
        self,
        sensor_frame,
        epochs: int = 8,
        scenarios_per_type: int = 2,
        learning_rate: float = 1e-3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        dataset = generate_scenario_dataset(sensor_frame, scenarios_per_type=scenarios_per_type)
        model = GridRiskGNN(
            node_feature_dim=len(NODE_FEATURE_COLUMNS),
            edge_feature_dim=len(EDGE_FEATURE_COLUMNS),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        last_loss = 0.0
        for _ in range(epochs):
            model.train()
            epoch_loss = 0.0
            for sample in dataset:
                optimizer.zero_grad()
                output = model(sample.node_features, sample.edge_index, sample.edge_features)
                loss = criterion(output.node_risk_score, sample.node_targets[:, 0])
                loss = loss + criterion(output.restoration_priority, sample.node_targets[:, 1])
                loss = loss + criterion(output.line_risk_score, sample.edge_targets[:, 0])
                loss = loss + criterion(output.overload_risk, sample.edge_targets[:, 1])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            last_loss = epoch_loss / max(len(dataset), 1)

        artifact = {
            "state_dict": model.state_dict(),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "node_feature_columns": list(NODE_FEATURE_COLUMNS),
            "edge_feature_columns": list(EDGE_FEATURE_COLUMNS),
            "scenario_count": len(dataset),
            "seed": seed,
        }
        self._save_artifact(artifact)
        return {
            "artifact_path": str(self.model_path),
            "train_graphs": len(dataset),
            "epochs": epochs,
            "final_loss": round(last_loss, 6),
        }

    def load_artifact(self) -> dict[str, Any]:
        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError("Grid risk GNN artifact not found.")
        return torch.load(self.model_path, map_location="cpu")

    def build_model(self) -> GridRiskGNN:
        artifact = self.load_artifact()
        model = GridRiskGNN(
            node_feature_dim=len(artifact["node_feature_columns"]),
            edge_feature_dim=len(artifact["edge_feature_columns"]),
            hidden_dim=int(artifact["hidden_dim"]),
            num_layers=int(artifact["num_layers"]),
        )
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        return model

    def _save_artifact(self, artifact: dict[str, Any]) -> None:
        assert self.model_path is not None
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, self.model_path)
