from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
