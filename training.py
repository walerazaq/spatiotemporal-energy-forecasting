import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Training

def train(model, train_loader, val_loader, device):
    model = model.to(device)

    loss_function = nn.HuberLoss(delta=100.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)

    # Initialize tracking metrics
    best_val_loss = float("inf")
    best_model = None
    epochs = 500

    print('-' * 30)
    print('Training ...')
    early_stop = 30
    es_counter = 0

    for epoch in range(epochs):
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_train_loss = 0

        for i, data in enumerate(tqdm(train_loader)):
            batch = data.batch.to(device)
            x = data.x.to(device)
            y = data.y.to(device)
            adj = data.adj.to(device)

            optimizer.zero_grad()
            out = model(x, adj, batch)

            step_loss = loss_function(out, y)
            step_loss.backward()
            optimizer.step()
            epoch_train_loss += step_loss.item()

        epoch_train_loss /= (i + 1)
        lr_scheduler.step()

        # Validate model
        val_loss, mse, rmse, mae = validate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1} train loss: {epoch_train_loss:.4f}, val MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metric_epoch = epoch + 1
            best_model = deepcopy(model)
            print("Saved new best model")
            es_counter = 0
        else:
            es_counter += 1

        if es_counter > early_stop:
            print('Early stopping, no improvement in validation loss.')
            break

    print(f"Training completed, best_val_loss: {best_val_loss:.4f}")
    return best_model


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    loss_func = nn.HuberLoss(delta=100.0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            y = data.y.to(device)
            adj = data.adj.to(device)

            out = model(x, adj, batch)
            step_loss = loss_func(out, y)
            val_loss += step_loss.item()

            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)

    avg_val_loss = val_loss / (i + 1)

    return avg_val_loss, mse, rmse, mae


def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            y = data.y.to(device)
            adj = data.adj.to(device)

            out = model(x, adj, batch)
            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)

    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return mse, rmse, mae, all_preds, all_labels
