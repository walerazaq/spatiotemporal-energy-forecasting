from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import (
    EdgeConv,
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


def graph_readout(x: torch.Tensor, method: str, batch: torch.Tensor) -> torch.Tensor:
    if method == "mean":
        return global_mean_pool(x, batch)
    if method == "meanmax":
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        return torch.cat((x_mean, x_max), dim=1)
    if method == "sum":
        return global_add_pool(x, batch)

    raise ValueError(f"Undefined readout operation: {method}")


def _coerce_edge_inputs(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not getattr(edge_index, "is_sparse", False):
        return edge_index, edge_weight

    coo = edge_index.to_sparse_coo().coalesce()
    return coo.indices().long(), coo.values().float()


class GraphTemporalModel(nn.Module):
    """EdgeConv + GCN spatial encoder followed by an LSTM temporal encoder."""

    def __init__(
        self,
        horizon: int,
        num_nodes: int,
        in_features: int,
        gcn_out_features: int,
        tr_hidden_size: int,
        mlp_hidden_size: int,
        readout: str = "meanmax",
    ):
        super().__init__()
        if readout not in {"mean", "meanmax", "sum"}:
            raise ValueError(f"Unsupported readout: {readout}")

        readout_multiplier = 2 if readout == "meanmax" else 1
        self.num_nodes = num_nodes
        self.readout = readout

        self.edgeconv = EdgeConv(
            nn.Sequential(
                nn.Linear(in_features * 2, in_features * 2),
                nn.ReLU(),
                nn.Linear(in_features * 2, gcn_out_features),
            )
        )
        self.gcn = GCNConv(gcn_out_features, gcn_out_features)
        self.lstm = nn.LSTM(
            input_size=gcn_out_features * readout_multiplier,
            hidden_size=tr_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(tr_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, horizon),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_index, edge_weight = _coerce_edge_inputs(edge_index, edge_weight)
        _, seq_len, _ = x.shape
        spatial_steps = []

        for step in range(seq_len):
            node_features = x[:, step, :]
            node_features = self.edgeconv(node_features, edge_index)
            node_features = self.gcn(
                node_features,
                edge_index,
                edge_weight=edge_weight,
            )
            spatial_steps.append(graph_readout(node_features, self.readout, batch))

        sequence = torch.stack(spatial_steps, dim=0).permute(1, 0, 2)
        sequence, _ = self.lstm(sequence)
        output = self.mlp(sequence[:, -1, :])
        return output.reshape(-1)


graphTS_model = GraphTemporalModel

