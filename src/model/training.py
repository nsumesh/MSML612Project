"""
Training loop for the LapTimeTransformer.

Uses Adam with a small weight decay and MSE loss on the last 5 output positions (the
prediction horizon). Gradients are clipped to 1.0 to keep training stable. A
ReduceLROnPlateau scheduler halves the learning rate when validation loss plateaus for
3 epochs, and early stopping kicks in after 8 epochs without improvement. The best
checkpoint (lowest validation loss) gets saved to models/{run_name}.pth, and the full
per-epoch loss and MAE history is written to results/{run_name}_history.csv.

Returns a summary dict so you can easily log results across multiple runs.
"""

import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from src.model.transformer import LapTimeTransformer
from src.data.dataset_file import get_dataloaders


def train_transformer(epochs=150, lr=1e-3, batch_size=64, weight_decay=1e-4, loss_name="mse", scheduler_patience=3, scheduler_factor=0.5, early_stop_patience=8, horizon=5, run_name="best_model", model_kwargs=None, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader, validation_loader, test_loader = get_dataloaders(batch_size=batch_size)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    info = json.loads(Path("data/splits/info.json").read_text())
    model_kwargs = {**info, **(model_kwargs or {})}
    model = LapTimeTransformer(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience)
    ckpt_path = Path("models") / f"{run_name}.pth"
    history_path = Path("results") / f"{run_name}_history.csv"

    best_val = float("inf")
    epochs_without_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        for data, label in training_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            prediction = output[:, -horizon:]
            loss = loss_function(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            train_mae_sum += (prediction - label).abs().mean().item()

        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        with torch.no_grad():
            for data, label in validation_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                prediction = output[:, -horizon:]
                val_loss_sum += loss_function(prediction, label).item()
                val_mae_sum += (prediction - label).abs().mean().item()

        train_loss = train_loss_sum / len(training_loader)
        val_loss = val_loss_sum / len(validation_loader)
        train_mae = train_mae_sum / len(training_loader)
        val_mae = val_mae_sum / len(validation_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append((epoch, train_loss, val_loss, train_mae, val_mae, current_lr))

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improve += 1

        if verbose:
            flag = " *" if improved else ""
            print(f"Epoch {epoch:3d} | Training Loss : {train_loss:.4f} (Mean Absolute Error: {train_mae:.3f}) "f"| Validation Loss : {val_loss:.4f} (Mean Absolute Error: {val_mae:.3f}) | Learning Rate : {current_lr:.2e}{flag}")

        if epochs_without_improve >= early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_mae", "val_mae", "lr"])
        writer.writerows(history)

    return {"run_name": run_name,"best_val_loss": best_val,"best_val_mae": min(h[4] for h in history),"epochs_run": len(history),"ckpt_path": str(ckpt_path),"history_path": str(history_path)}

if __name__ == "__main__":
    train_transformer(epochs=50, lr=1e-3, run_name="best_model")
