import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from src.model.transformer import LapTimeTransformer


COMPOUND_COLORS = {
    "SOFT": "#E8002D",
    "MEDIUM": "#F5A623",
    "HARD": "#B0B0B0",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
}


def plot_race_degradation(race="Monaco", year=2025, driver="LEC",
                          model_path="models/best_model.pth",
                          raw_csv="data/raw/all_laps.csv",
                          meta_path="data/splits/meta_test.csv",
                          horizon=5, lookback=15):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LapTimeTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load full raw race data (includes pit laps for the degradation drop)
    raw = pd.read_csv(raw_csv)
    race_data = raw[(raw["Race"] == race) & (raw["Year"] == year) & (raw["Driver"] == driver)]
    race_data = race_data.sort_values("LapNumber").reset_index(drop=True)

    if race_data.empty:
        print(f"No data found for {driver} at {race} {year}")
        return

    # Load test metadata to find windows for this driver/race
    meta = pd.read_csv(meta_path)
    test_data = np.load("data/splits/data_test.npy")
    test_labels = np.load("data/splits/labels_test.npy")

    mask = (meta["race"] == race) & (meta["year"] == year) & (meta["driver"] == driver)
    indices = np.where(mask.values)[0]

    if len(indices) == 0:
        # Find the driver/race combo with most windows in the test set
        best = meta.groupby(["year", "race", "driver"]).size().idxmax()
        year, race, driver = best
        print(f"No windows found for requested combo — using {driver} | {race} {year} instead")
        race_data = raw[(raw["Race"] == race) & (raw["Year"] == year) & (raw["Driver"] == driver)]
        race_data = race_data.sort_values("LapNumber").reset_index(drop=True)
        mask = (meta["race"] == race) & (meta["year"] == year) & (meta["driver"] == driver)
        indices = np.where(mask.values)[0]

    # Run model on matched windows
    pred_laps, pred_times, actual_times = [], [], []
    if len(indices) > 0:
        X = torch.tensor(test_data[indices], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(X)
            preds = out[:, -horizon:].cpu().numpy()
        labels = test_labels[indices]
        start_laps = meta.loc[mask, "start_lap"].values

        seen_laps = set()
        for w_idx in range(len(indices)):
            for h in range(horizon):
                lap_num = start_laps[w_idx] + h
                if lap_num not in seen_laps:
                    pred_laps.append(lap_num)
                    pred_times.append(preds[w_idx, h])
                    actual_times.append(labels[w_idx, h])
                    seen_laps.add(lap_num)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot full actual race — coloured by compound, drops show pit stops
    for _, row in race_data.iterrows():
        if pd.isna(row["LapTime"]) or pd.isna(row["Compound"]):
            continue
        color = COMPOUND_COLORS.get(row["Compound"], "gray")
        ax.scatter(row["LapNumber"], row["LapTime"], color=color, s=18, zorder=3)

    # Connect the dots per compound stint
    ax.plot(race_data["LapNumber"], race_data["LapTime"],
            color="#444444", linewidth=0.8, zorder=2, label="_nolegend_")

    # Overlay model predictions
    if pred_laps:
        ax.scatter(pred_laps, actual_times, color="dodgerblue", marker="o", s=35,
                   zorder=4, label="Actual (clean laps)", alpha=0.9, edgecolors="white", linewidths=0.5)
        ax.scatter(pred_laps, pred_times, color="black", marker="D", s=35,
                   zorder=5, label="Model Prediction", edgecolors="white", linewidths=0.5)

    # Compound legend patches
    compounds_used = race_data["Compound"].dropna().unique()
    compound_patches = [
        mpatches.Patch(color=COMPOUND_COLORS.get(c, "gray"), label=c)
        for c in compounds_used
    ]
    first_legend = ax.legend(handles=compound_patches, title="Tyre Compound",
                             loc="upper left", framealpha=0.9)
    ax.add_artist(first_legend)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(f"Tyre Degradation & Lap Time Prediction — {driver} | {race} {year}")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#444444")
    ax.grid(alpha=0.15, color="white")

    Path("results").mkdir(exist_ok=True)
    out_path = f"results/{driver}_{race}_{year}_degradation.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    # Change these to any driver/race in the test set
    plot_race_degradation(race="Monaco", year=2025, driver="LEC")
