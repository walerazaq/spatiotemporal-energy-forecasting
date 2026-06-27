from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def calculate_activity_adjacency(window_data: pd.DataFrame) -> np.ndarray:
    """Calculate simultaneous machine activity adjacency for a data window."""
    binary_matrix = (window_data > 0).astype(int)
    n_machines = binary_matrix.shape[1]
    adj_matrix = np.zeros((n_machines, n_machines), dtype=np.float32)

    for source in range(n_machines):
        source_active = binary_matrix.iloc[:, source] == 1
        total_source_active = np.sum(source_active)
        if total_source_active == 0:
            continue

        for target in range(n_machines):
            both_active = np.sum(source_active & (binary_matrix.iloc[:, target] == 1))
            adj_matrix[source, target] = both_active / total_source_active

    return adj_matrix


def adjacency_to_edge_tensors(
    adj_matrix: np.ndarray,
    *,
    fallback_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense adjacency matrix to PyG edge_index and edge_weight tensors."""
    sources, targets = np.nonzero(adj_matrix > 0)

    if len(sources) == 0 and fallback_self_loops:
        node_count = adj_matrix.shape[0]
        sources = np.arange(node_count)
        targets = np.arange(node_count)
        weights = np.ones(node_count, dtype=np.float32)
    else:
        weights = adj_matrix[sources, targets].astype(np.float32)

    edge_index = torch.tensor(np.vstack([sources, targets]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight
