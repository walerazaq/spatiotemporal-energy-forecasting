from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .graph import adjacency_to_edge_tensors, calculate_activity_adjacency


FEATURE_SUFFIXES = {
    "P_kW": "_P_kW",
    "S_kVA": "_S_kVA",
    "std_P_kW": "_std_P_kW",
    "std_S_kVA": "_std_S_kVA",
}


def _read_numeric_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file does not exist: {path}")

    df = pd.read_csv(path, index_col=0)
    return df.apply(pd.to_numeric, errors="coerce")


def _machine_names(columns: pd.Index, feature_name: str) -> list[str]:
    suffix = FEATURE_SUFFIXES[feature_name]
    return [
        column[: -len(suffix)] if column.endswith(suffix) else column
        for column in columns
    ]


class EnergyGraphDataset(torch.utils.data.Dataset):
    """Windowed PyG graph dataset for HIPE energy forecasting."""

    feature_files = {
        "P_kW": "P_kW.csv",
        "S_kVA": "S_kVA.csv",
        "std_P_kW": "std_P_kW.csv",
        "std_S_kVA": "std_S_kVA.csv",
    }

    def __init__(
        self,
        folder_path: str | os.PathLike[str],
        horizon: int,
        window_size: int = 12,
        stride: int = 3,
        *,
        target_file: str = "total_power.csv",
        target_column: str | None = None,
    ):
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if stride < 1:
            raise ValueError("stride must be at least 1")

        self.folder_path = Path(folder_path)
        self.horizon = horizon
        self.window_size = window_size
        self.stride = stride
        self.target_file = target_file
        self.target_column = target_column
        self.files = {
            name: self.folder_path / filename
            for name, filename in self.feature_files.items()
        }
        self.files["target"] = self.folder_path / target_file
        self.data = self.load_data()

    def load_data(self) -> list[Data]:
        features = {
            name: _read_numeric_csv(path)
            for name, path in self.files.items()
            if name != "target"
        }
        target = _read_numeric_csv(self.files["target"])

        if self.target_column is not None:
            if self.target_column not in target.columns:
                raise ValueError(
                    f"Target column {self.target_column!r} is not in {self.files['target']}"
                )
            target = target[[self.target_column]]
        elif len(target.columns) != 1:
            raise ValueError(
                "Target CSV must have exactly one target column or target_column must be set"
            )

        self._validate_feature_tables(features, target)

        feature_list = [
            features[name].to_numpy(dtype=np.float32)
            for name in self.feature_files
        ]
        features_array = np.stack(feature_list, axis=-1)
        features_array = np.nan_to_num(features_array, nan=0.0)
        target_array = target.to_numpy(dtype=np.float32)
        target_array = np.nan_to_num(target_array, nan=0.0)

        end = len(features_array) - self.window_size - self.horizon + 1
        if end <= 0:
            return []

        data = []
        for i in range(0, end, self.stride):
            window_data = features["P_kW"].iloc[i : i + self.window_size]
            adj_matrix = calculate_activity_adjacency(window_data)
            edge_index, edge_weight = adjacency_to_edge_tensors(adj_matrix)

            x_tensor = torch.tensor(
                features_array[i : i + self.window_size],
                dtype=torch.float32,
            ).permute(1, 0, 2)

            target_start = i + self.window_size
            target_end = target_start + self.horizon
            y_tensor = torch.tensor(
                target_array[target_start:target_end],
                dtype=torch.float32,
            ).reshape(-1)

            data.append(
                Data(
                    x=x_tensor,
                    y=y_tensor,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    num_nodes=x_tensor.size(0),
                )
            )

        return data

    def _validate_feature_tables(
        self,
        features: dict[str, pd.DataFrame],
        target: pd.DataFrame,
    ) -> None:
        for key, df in features.items():
            if not df.index.equals(target.index):
                raise ValueError(f"Timestamp mismatch between {key} and target")

        machine_order = _machine_names(features["P_kW"].columns, "P_kW")
        for feature_name, df in features.items():
            names = _machine_names(df.columns, feature_name)
            if names != machine_order:
                raise ValueError(
                    f"Machine column order mismatch in {feature_name}; "
                    "all feature files must list machines in the same order"
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
