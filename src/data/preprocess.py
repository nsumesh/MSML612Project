import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


LAP_TIME_MIN = 60
LAP_TIME_MAX = 200


COMPOUND_MAP = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "WET": 3,
    "INTERMEDIATE": 4
}

def clean_laps(laps : pd.DataFrame):
    laps = laps.dropna(subset=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "SpeedST", "TyreLife", "Compound"])
    laps = laps[(laps["LapTime"] >= LAP_TIME_MIN) & (laps["LapTime"] <= LAP_TIME_MAX)]
    laps = laps[laps["LapNumber"] > 1]
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    laps["CompoundEncoded"] = laps["Compound"].map(COMPOUND_MAP).fillna(2).astype(int)
    drivers = laps["Driver"].unique()
    driver_to_id = {d: i for i, d in enumerate(sorted(drivers))}
    laps["DriverID"] = laps["Driver"].map(driver_to_id)
    return laps.reset_index(drop=True)


def _windows_from_laps(group, features, target, lookback, horizon):
    """Create sliding windows from a contiguous block of laps."""
    windows_data, windows_labels = [], []
    for i in range(len(group) - lookback - horizon + 1):
        windows_data.append(group[features].iloc[i:i + lookback].values)
        windows_labels.append(group[target].iloc[i + lookback:i + lookback + horizon].values)
    return windows_data, windows_labels


def build_transformer_sequences(laps, lookback=15, horizon=5,
                                train_frac=0.7, val_frac=0.15):
    """
    Assign entire driver/race groups to train/val/test, then create windows.
    This prevents leakage from overlapping windows across splits.
    """
    features = [
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "SpeedST",
        "TyreLife",
        "CompoundEncoded",
    ]
    target = "LapTime"

    # Collect all valid groups
    rng = np.random.default_rng(42)
    groups = []
    for (race, year, driver), group in laps.groupby(["Race", "Year", "Driver"]):
        group = group.sort_values("LapNumber").reset_index(drop=True)
        if len(group) >= lookback + horizon:
            groups.append(group)
    rng.shuffle(groups)

    # Assign groups to splits
    n_groups = len(groups)
    train_end = int(train_frac * n_groups)
    val_end = int((train_frac + val_frac) * n_groups)

    group_splits = {
        "train": groups[:train_end],
        "val":   groups[train_end:val_end],
        "test":  groups[val_end:],
    }

    splits = {k: ([], [], []) for k in ("train", "val", "test")}
    for split_name, split_groups in group_splits.items():
        for group in split_groups:
            race   = group["Race"].iloc[0]
            year   = group["Year"].iloc[0]
            driver = group["Driver"].iloc[0]
            w_data, w_labels = _windows_from_laps(group, features, target, lookback, horizon)
            splits[split_name][0].extend(w_data)
            splits[split_name][1].extend(w_labels)
            splits[split_name][2].extend([{"race": race, "year": year, "driver": driver}] * len(w_data))

    result = {}
    for split_name in ("train", "val", "test"):
        if splits[split_name][0]:
            result[split_name] = (
                np.array(splits[split_name][0], dtype=np.float32),
                np.array(splits[split_name][1], dtype=np.float32),
                splits[split_name][2],
            )
        else:
            result[split_name] = (
                np.empty((0, lookback, len(features)), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32),
                [],
            )
    return result


def run_preprocessing(csv_path="data/raw/all_laps.csv", save_dir="data/splits/"):
    laps = pd.read_csv(csv_path)
    laps = clean_laps(laps)
    splits = build_transformer_sequences(laps)
    return normalize_and_save(splits, save_dir)


def normalize_and_save(splits, save_dir="data/splits/"):
    """Fit scaler on train data only, then transform all splits."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    train_data, train_labels, _ = splits["train"]
    n, seq, feat = train_data.shape

    # Fit scaler on training data only to avoid leaking test statistics
    scaler = StandardScaler()
    scaler.fit(train_data.reshape(-1, feat))

    for split_name in ("train", "val", "test"):
        data, labels, metadata = splits[split_name]
        if len(data) > 0:
            n_s, seq_s, feat_s = data.shape
            data_scaled = scaler.transform(data.reshape(-1, feat_s)).reshape(n_s, seq_s, feat_s)
        else:
            data_scaled = data
        np.save(f"{save_dir}/data_{split_name}.npy", data_scaled)
        np.save(f"{save_dir}/labels_{split_name}.npy", labels)
        pd.DataFrame(metadata).to_csv(f"{save_dir}/meta_{split_name}.csv", index=False)
        print(f"  {split_name}: {len(data)} samples")

    with open(f"{save_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return scaler
