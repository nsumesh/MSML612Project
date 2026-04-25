import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import torch
from src.model.transformer import LapTimeTransformer
from src.data.dataset_file import get_dataloaders

def setup_ax(ax):
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)

def load_predictions(model_path="models/best_model.pth",
                     meta_path="data/splits/meta_test.csv",
                     horizon=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=batch_size)
    model = LapTimeTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, label in test_loader:
            out = model(data.to(device))
            all_preds.append(out[:, -horizon:].cpu().numpy())
            all_labels.append(label.numpy())

    preds  = np.concatenate(all_preds,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    meta   = pd.read_csv(meta_path).iloc[:len(preds)].copy()
    meta["mae"] = np.abs(preds - labels).mean(axis=1)
    return preds, labels, meta


def plot_horizon_mae(preds, labels, save_dir):
    horizon = preds.shape[1]
    mae_per_lap = [np.abs(preds[:, i] - labels[:, i]).mean() for i in range(horizon)]

    fig, ax = plt.subplots(figsize=(7, 4))
    setup_ax(ax)

    bars = ax.bar([f"Lap +{i+1}" for i in range(horizon)], mae_per_lap,
                  color="steelblue", edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, mae_per_lap):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(mae_per_lap) * 1.25)
    ax.set_xlabel("Predicted Lap")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("MAE per Predicted Lap (Horizon Breakdown)")

    plt.tight_layout()
    path = save_dir / "horizon_mae.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_race_mae(meta, save_dir):
    race_mae = (
        meta.groupby(["year", "race"])["mae"]
        .mean()
        .reset_index()
        .assign(label=lambda d: d["race"] + " (" + d["year"].astype(str) + ")")
        .sort_values("mae")
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    setup_ax(ax)

    bars = ax.barh(race_mae["label"], race_mae["mae"], color="steelblue",
                   edgecolor="white", linewidth=0.5, height=0.6)
    for bar, val in zip(bars, race_mae["mae"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s", va="center", fontsize=7.5)

    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="0.5s target")
    ax.set_xlabel("MAE (seconds)")
    ax.set_title("MAE by Race")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = save_dir / "race_mae.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_driver_mae(meta, save_dir):
    driver_mae = (
        meta.groupby("driver")["mae"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "MAE", "count": "windows"})
        .sort_values("MAE")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    setup_ax(ax)

    bars = ax.bar(driver_mae["driver"], driver_mae["MAE"], color="steelblue",
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, driver_mae["MAE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="0.5s target")
    ax.set_xlabel("Driver")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("MAE by Driver")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    path = save_dir / "driver_mae.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    save_dir = Path("plots")
    save_dir.mkdir(exist_ok=True)

    preds, labels, meta = load_predictions()
    plot_horizon_mae(preds, labels, save_dir)
    plot_race_mae(meta, save_dir)
    plot_driver_mae(meta, save_dir)
    print("All plots saved to plots/")
