from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_predictions(pred, act, start: int = 0, end: int | None = None, *, save_path=None):
    if end is None or end > len(pred):
        end = len(pred)

    pred_range = pred[start:end]
    act_range = act[start:end]

    plt.figure(figsize=(15, 6))
    plt.plot(act_range, label="Actual", color="b", linewidth=2)
    plt.plot(pred_range, label="Predicted", color="r", linestyle="--", linewidth=2)
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted Values")
    plt.legend()

    if save_path is not None:
        plt.savefig(Path(save_path), bbox_inches="tight")
    else:
        plt.show()

