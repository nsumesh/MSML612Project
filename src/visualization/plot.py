import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.model.transformer import LapTimeTransformer
from src.data.dataset_file import get_dataloaders


def plot_predictions(model_path="models/best_model.pth", horizon=5, n_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(batch_size=64)

    model = LapTimeTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_inputs = [], [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            output = model(data)
            preds = output[:, -horizon:]
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.numpy())
            all_inputs.append(data.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)[:n_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:n_samples]
    all_inputs = np.concatenate(all_inputs, axis=0)[:n_samples]

    # Plot 1: Actual vs Predicted across all test windows
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    flat_preds  = all_preds.reshape(-1)
    flat_labels = all_labels.reshape(-1)
    x = np.arange(len(flat_labels))
    ax.plot(x, flat_labels, label="Actual", alpha=0.7, linewidth=0.8)
    ax.plot(x, flat_preds,  label="Predicted", alpha=0.7, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Prediction index")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title("Actual vs Predicted Lap Times (Test Set)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Single race reconstruction — context + predicted vs actual
    ax2 = axes[1]
    sample_idx = 0
    context_laps = all_inputs[sample_idx, :, 0]   # LapTime is feature index 0
    actual_next  = all_labels[sample_idx]
    pred_next    = all_preds[sample_idx]

    lookback = len(context_laps)
    context_x = np.arange(lookback)
    future_x  = np.arange(lookback, lookback + horizon)

    ax2.plot(context_x, context_laps, color="steelblue", label="Context (scaled)", linewidth=1.2)
    ax2.plot(future_x, actual_next, color="green", marker="o", label="Actual next laps", linewidth=1.2)
    ax2.plot(future_x, pred_next,   color="red",   marker="x", label="Predicted next laps", linewidth=1.2, linestyle="--")
    ax2.axvline(x=lookback - 0.5, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Lap (relative)")
    ax2.set_ylabel("Lap Time")
    ax2.set_title("Single Window: Context → Predicted vs Actual")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/lap_time_predictions.png", dpi=150)
    print("Saved to results/lap_time_predictions.png")
    plt.show()


if __name__ == "__main__":
    import pathlib
    pathlib.Path("results").mkdir(exist_ok=True)
    plot_predictions()
