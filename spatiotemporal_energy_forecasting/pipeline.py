from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .dataset import EnergyGraphDataset
from .model import GraphTemporalModel
from .training import test_model, train
from .visualization import plot_predictions


@dataclass
class TrainingConfig:
    data_dir: str | Path
    horizon: int = 1
    window_size: int = 12
    stride: int = 1
    batch_size: int = 32
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "auto"
    epochs: int = 500
    early_stop: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gcn_out_features: int = 8
    tr_hidden_size: int = 16
    mlp_hidden_size: int = 32
    readout: str = "meanmax"
    seed: int | None = None


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def split_dataset(dataset, train_ratio: float = 0.6, val_ratio: float = 0.2):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size : train_size + val_size]
    test_dataset = dataset[train_size + val_size :]

    if not train_dataset or not val_dataset or not test_dataset:
        raise ValueError(
            "Dataset split produced an empty train, validation, or test partition"
        )

    return train_dataset, val_dataset, test_dataset


def build_data_loaders(config: TrainingConfig, device: torch.device):
    dataset = EnergyGraphDataset(
        config.data_dir,
        config.horizon,
        window_size=config.window_size,
        stride=config.stride,
    )
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    pin_memory = config.pin_memory and device.type == "cuda"
    persistent_workers = config.num_workers > 0
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        dataset,
    )


def run_training(config: TrainingConfig, *, plot: bool = False):
    if config.seed is not None:
        torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    train_loader, val_loader, test_loader, dataset = build_data_loaders(
        config,
        device,
    )
    sample = dataset[0]
    model = GraphTemporalModel(
        config.horizon,
        num_nodes=sample.num_nodes,
        in_features=sample.x.size(-1),
        gcn_out_features=config.gcn_out_features,
        tr_hidden_size=config.tr_hidden_size,
        mlp_hidden_size=config.mlp_hidden_size,
        readout=config.readout,
    )

    trained_model = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=config.epochs,
        early_stop=config.early_stop,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    mse, rmse, mae, pred, act = test_model(trained_model, test_loader, device)
    if plot:
        plot_predictions(pred, act)

    return {
        "model": trained_model,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "predictions": pred,
        "actuals": act,
    }

