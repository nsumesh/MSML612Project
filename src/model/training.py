import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from src.model.transformer import LapTimeTransformer
from src.data.dataset_file import get_dataloaders


LOSS_FUNCTIONS = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "l1": nn.L1Loss,
    "smoothl1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
}


def train_transformer(
    epochs=150,
    lr=1e-3,
    batch_size=64,
    weight_decay=1e-4,
    loss_name="mse",
    scheduler_patience=3,
    scheduler_factor=0.5,
    early_stop_patience=8,
    horizon=5,
    run_name="best_model",
    model_kwargs=None,
    verbose=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader, validation_loader, _ = get_dataloaders(batch_size=batch_size)

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    model_kwargs = model_kwargs or {}
    model = LapTimeTransformer(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = LOSS_FUNCTIONS[loss_name.lower()]()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

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
            print(
                f"Epoch {epoch:3d} | train {train_loss:.4f} (mae {train_mae:.3f}) "
                f"| val {val_loss:.4f} (mae {val_mae:.3f}) | lr {current_lr:.2e}{flag}"
            )

        if epochs_without_improve >= early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_mae", "val_mae", "lr"])
        writer.writerows(history)

    return {
        "run_name": run_name,
        "best_val_loss": best_val,
        "best_val_mae": min(h[4] for h in history),
        "epochs_run": len(history),
        "ckpt_path": str(ckpt_path),
        "history_path": str(history_path),
    }


if __name__ == "__main__":
    train_transformer(epochs=50, lr=1e-3, run_name="best_model")
