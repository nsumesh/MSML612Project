"""
Takes the raw lap CSV and turns it into model-ready train/val/test splits.

There are three main steps. First, clean_laps drops rows with missing values, filters
out anything with a lap time outside 60–200 seconds, removes lap 1 (always an outlier),
encodes tyre compound as an integer, figures out each driver's stint number and previous
compound through the race, and removes any lap where a pit stop was happening.

Second, build_transformer_sequences groups the clean laps by driver-race pair, shuffles
those groups (seeded for reproducibility), and splits them 70/15/15 into train/val/test
at the group level — so no single driver's race straddles two splits. It then slides a
window of 15 context laps followed by 5 label laps across each group. Labels are stored
as residuals (difference from the last context lap time) to make the learning problem easier.

Finally, normalize_and_save fits a StandardScaler on the 12 numerical features of the
training data only, applies it to all splits, and saves the .npy arrays, metadata CSVs,
the scaler, and a small info.json with driver and track counts.
"""

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


minimum_lap_time = 60
max_lap_time = 200

numerical_features = 12

compound_map = {"SOFT": 0,"MEDIUM": 1,"HARD": 2,"WET": 3,"INTERMEDIATE": 4}


def add_stint_features(laps: pd.DataFrame) -> pd.DataFrame:
    laps = laps.copy()
    laps["StintNumber"] = 1
    laps["PrevCompound"] = laps["CompoundEncoded"]

    for (_, _, _), group in laps.groupby(["Race", "Year", "Driver"]):
        group = group.sort_values("LapNumber")
        stint = 1
        prev_comp = int(group["CompoundEncoded"].iloc[0])

        for row_idx, row in group.iterrows():
            laps.loc[row_idx, "StintNumber"] = stint
            laps.loc[row_idx, "PrevCompound"] = prev_comp
            if pd.notna(row["PitInTime"]):
                prev_comp = int(row["CompoundEncoded"])
                stint += 1
    return laps


def clean_laps(laps: pd.DataFrame):
    laps = laps.dropna(subset=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "SpeedST", "TyreLife", "Compound", "AirTemp", "TrackTemp", "Rainfall"])
    laps = laps[(laps["LapTime"] >= minimum_lap_time) & (laps["LapTime"] <= max_lap_time)]
    laps = laps[laps["LapNumber"] > 1]
    laps["CompoundEncoded"] = laps["Compound"].map(compound_map).fillna(2).astype(int)
    laps = add_stint_features(laps)
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    drivers = laps["Driver"].unique()
    driver_to_id = {d: i for i, d in enumerate(sorted(drivers))}
    laps["DriverID"] = laps["Driver"].map(driver_to_id)
    tracks = sorted(laps["Race"].unique())
    track_to_id = {t: i for i, t in enumerate(tracks)}
    laps["TrackID"] = laps["Race"].map(track_to_id)
    return laps.reset_index(drop=True), driver_to_id, track_to_id


def windows_from_laps(group, features, target, lookback, horizon, residual=True):
    windows_data, windows_labels, start_laps, last_lap_times = [], [], [], []
    for i in range(len(group) - lookback - horizon + 1):
        x = group[features].iloc[i:i + lookback].values
        y = group[target].iloc[i + lookback:i + lookback + horizon].values
        last_lap = float(group[target].iloc[i + lookback - 1])
        if residual:
            y = y - last_lap
        windows_data.append(x)
        windows_labels.append(y)
        start_laps.append(int(group["LapNumber"].iloc[i + lookback]))
        last_lap_times.append(last_lap)
    return windows_data, windows_labels, start_laps, last_lap_times


def build_transformer_sequences(laps, lookback=15, horizon=5):
    numerical_features = ["LapTime","Sector1Time","Sector2Time","Sector3Time","SpeedST","TyreLife","CompoundEncoded","AirTemp","TrackTemp","Rainfall","StintNumber","PrevCompound"]
    id_features = ["TrackID", "DriverID"]
    features = numerical_features + id_features
    target = "LapTime"
    rng = np.random.default_rng(42)
    groups = []
    for (race, year, driver), group in laps.groupby(["Race", "Year", "Driver"]):
        group = group.sort_values("LapNumber").reset_index(drop=True)
        if len(group) >= lookback + horizon:
            groups.append(group)
    rng.shuffle(groups)
    # Assign groups to splits
    n_groups = len(groups)
    train_end = int(0.7 * n_groups)
    val_end = int((0.85) * n_groups)
    group_splits = {"train": groups[:train_end],"val": groups[train_end:val_end],"test": groups[val_end:]}
    splits = {k: ([], [], []) for k in ("train", "val", "test")}
    for split_name, split_groups in group_splits.items():
        for group in split_groups:
            race = group["Race"].iloc[0]
            year = group["Year"].iloc[0]
            driver = group["Driver"].iloc[0]
            w_data, w_labels, w_start_laps, w_last_laps = windows_from_laps(group, features, target, lookback, horizon)
            splits[split_name][0].extend(w_data)
            splits[split_name][1].extend(w_labels)
            splits[split_name][2].extend([{"race": race, "year": year, "driver": driver, "start_lap": sl, "last_lap_time": llt} for sl, llt in zip(w_start_laps, w_last_laps)])
    result = {}
    for split_name in ("train", "val", "test"):
        if splits[split_name][0]:
            result[split_name] = (np.array(splits[split_name][0], dtype=np.float32),np.array(splits[split_name][1], dtype=np.float32),splits[split_name][2],)
        else:
            result[split_name] = (np.empty((0, lookback, len(features)), dtype=np.float32),np.empty((0, horizon), dtype=np.float32),[])
    return result


def run_preprocessing(csv_path="data/raw/all_laps.csv", save_dir="data/splits/"):
    laps = pd.read_csv(csv_path)
    laps, driver_to_id, track_to_id = clean_laps(laps)
    n_drivers = int(laps["DriverID"].max()) + 1
    n_tracks  = int(laps["TrackID"].max())  + 1
    splits = build_transformer_sequences(laps)
    return normalize_and_save(splits, save_dir, n_drivers=n_drivers, n_tracks=n_tracks, driver_to_id=driver_to_id, track_to_id=track_to_id)


def normalize_and_save(splits, save_dir="data/splits/", n_drivers=30, n_tracks=30, driver_to_id=None, track_to_id=None):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    train_data, _, _ = splits["train"]
    feat = train_data.shape[2]
    scaler = StandardScaler()
    scaler.fit(train_data.reshape(-1, feat)[:, :numerical_features])

    for split_name in ("train", "val", "test"):
        data, labels, metadata = splits[split_name]
        if len(data) > 0:
            n_s, seq_s, feat_s = data.shape
            flat = data.reshape(-1, feat_s)
            flat_scaled = flat.copy()
            flat_scaled[:, :numerical_features] = scaler.transform(flat[:, :numerical_features])
            data_out = flat_scaled.reshape(n_s, seq_s, feat_s)
        else:
            data_out = data
        np.save(f"{save_dir}/data_{split_name}.npy", data_out)
        np.save(f"{save_dir}/labels_{split_name}.npy", labels)
        pd.DataFrame(metadata).to_csv(f"{save_dir}/meta_{split_name}.csv", index=False)
        print(f"  {split_name}: {len(data)} samples")

    with open(f"{save_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    info = {"n_drivers": n_drivers, "n_tracks": n_tracks, "driver_to_id": driver_to_id, "track_to_id": track_to_id}
    with open(f"{save_dir}/info.json", "w") as f:
        json.dump(info, f)
    print(f" info: {info}")

    return scaler
