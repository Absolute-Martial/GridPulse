"""Pure PyTorch message-passing models for GridPulse topology risk."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class MessagePassingLayer(nn.Module):
    """Small undirected mean-aggregation message-passing block."""

    def __init__(self, hidden_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        edge_inputs = torch.cat([node_state[src], node_state[dst], edge_features], dim=-1)
        edge_messages = self.edge_mlp(edge_inputs)

        aggregate = torch.zeros_like(node_state)
        counts = torch.zeros(node_state.size(0), 1, device=node_state.device, dtype=node_state.dtype)
        aggregate.index_add_(0, src, edge_messages)
        aggregate.index_add_(0, dst, edge_messages)
        counts.index_add_(0, src, torch.ones_like(counts[src]))
        counts.index_add_(0, dst, torch.ones_like(counts[dst]))
        aggregate = aggregate / counts.clamp_min(1.0)

        updated = self.node_mlp(torch.cat([node_state, aggregate], dim=-1))
        return self.norm(node_state + updated)


@dataclass(frozen=True)
class GridRiskOutput:
    node_risk_score: torch.Tensor
    line_risk_score: torch.Tensor
    overload_risk: torch.Tensor
    restoration_priority: torch.Tensor


class GridRiskGNN(nn.Module):
    """Simple message-passing network with node and edge risk heads."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim=hidden_dim, edge_feature_dim=edge_feature_dim) for _ in range(num_layers)]
        )
        self.edge_readout = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_risk_head = nn.Linear(hidden_dim, 1)
        self.restoration_head = nn.Linear(hidden_dim, 1)
        self.line_risk_head = nn.Linear(hidden_dim, 1)
        self.overload_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> GridRiskOutput:
        node_state = self.node_encoder(node_features)
        for layer in self.layers:
            node_state = layer(node_state, edge_index, edge_features)

        src, dst = edge_index
        edge_state = self.edge_readout(torch.cat([node_state[src], node_state[dst], edge_features], dim=-1))

        return GridRiskOutput(
            node_risk_score=torch.sigmoid(self.node_risk_head(node_state)).squeeze(-1),
            line_risk_score=torch.sigmoid(self.line_risk_head(edge_state)).squeeze(-1),
            overload_risk=torch.sigmoid(self.overload_head(edge_state)).squeeze(-1),
            restoration_priority=torch.sigmoid(self.restoration_head(node_state)).squeeze(-1),
        )
