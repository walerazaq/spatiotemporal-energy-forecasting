from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


def _edge_weight(data, device: torch.device) -> torch.Tensor | None:
    edge_weight = getattr(data, "edge_weight", None)
    return None if edge_weight is None else edge_weight.to(device)


def _forward_batch(model, data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch = data.batch.to(device)
    x = data.x.to(device)
    y = data.y.to(device)

    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None:
        edge_index = edge_index.to(device)
        return model(x, edge_index, batch, _edge_weight(data, device)), y

    adj = getattr(data, "adj", None)
    if adj is not None:
        adj = adj.to(device)
        return model(x, adj, batch), y

    raise AttributeError("Batch data must contain edge_index or adj")


def train(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int = 500,
    early_stop: int = 30,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    huber_delta: float = 100.0,
    scheduler_t_max: int | None = 400,
    eta_min: float = 1e-5,
    verbose: bool = True,
):
    model = model.to(device)

    loss_function = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max or epochs,
        eta_min=eta_min,
    )

    best_val_loss = float("inf")
    best_model = None
    best_metric_epoch = 0
    early_stop_counter = 0

    if verbose:
        print("-" * 30)
        print("Training ...")

    for epoch in range(epochs):
        if verbose:
            print("-" * 30)
            print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        epoch_train_loss = 0.0
        batch_count = 0

        for data in tqdm(train_loader, disable=not verbose):
            optimizer.zero_grad()
            out, y = _forward_batch(model, data, device)
            step_loss = loss_function(out, y)
            step_loss.backward()
            optimizer.step()

            epoch_train_loss += step_loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("train_loader produced no batches")

        epoch_train_loss /= batch_count
        lr_scheduler.step()

        val_loss, mse, rmse, mae = validate_model(
            model,
            val_loader,
            device,
            huber_delta=huber_delta,
        )
        if verbose:
            print(
                f"Epoch {epoch + 1} train loss: {epoch_train_loss:.4f}, "
                f"val MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metric_epoch = epoch + 1
            best_model = deepcopy(model)
            early_stop_counter = 0
            if verbose:
                print("Saved new best model")
        else:
            early_stop_counter += 1

        if early_stop_counter > early_stop:
            if verbose:
                print("Early stopping, no improvement in validation loss.")
            break

    if best_model is None:
        raise RuntimeError("Training finished without producing a best model")

    if verbose:
        print(
            f"Training completed, best_val_loss: {best_val_loss:.4f} "
            f"at epoch {best_metric_epoch}"
        )
    return best_model


def validate_model(model, val_loader, device, *, huber_delta: float = 100.0):
    model.eval()
    val_loss = 0.0
    loss_func = nn.HuberLoss(delta=huber_delta)

    all_preds = []
    all_labels = []
    batch_count = 0

    with torch.no_grad():
        for data in val_loader:
            out, y = _forward_batch(model, data, device)
            step_loss = loss_func(out, y)
            val_loss += step_loss.item()
            batch_count += 1

            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    if batch_count == 0:
        raise ValueError("val_loader produced no batches")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    avg_val_loss = val_loss / batch_count

    return avg_val_loss, mse, rmse, mae


def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            out, y = _forward_batch(model, data, device)
            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    if not all_preds:
        raise ValueError("test_loader produced no batches")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)

    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return mse, rmse, mae, all_preds, all_labels
