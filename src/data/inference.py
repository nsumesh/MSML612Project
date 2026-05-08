"""
Online inference pipeline used by the Streamlit app at runtime.

prepare_windows fetches a live race session through FastF1 and applies the exact same
cleaning and feature engineering that was used during training — NaN filtering, lap time
bounds, compound encoding, stint tracking, pit lap removal. 

It then maps the driver and circuit to the integer IDs they were assigned during training, isolates the one driver
you asked for, and slides every possible 15-lap window across their race.

Each window gets scaled with the saved StandardScaler (numerical columns only; TrackID and DriverID
pass through as-is). Returns the tensor of windows ready for the model, a metadata list
(start lap, last lap time, compound, stint number for each window), and the full filtered
driver lap DataFrame so the app can plot context laps and actuals on the chart.
"""
import numpy as np
import torch

from src.data.fetch import fetch_race
from src.data.preprocess import (compound_map, minimum_lap_time, max_lap_time, add_stint_features, numerical_features)

lookback_context = 15
prediction_horizon = 5

features = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time","SpeedST", "TyreLife", "CompoundEncoded","AirTemp", "TrackTemp", "Rainfall","StintNumber", "PrevCompound",]
all_features = features + ["TrackID", "DriverID"]


def prepare_windows(year, race_name, driver, info, scaler):
    driver_to_id = info["driver_to_id"]
    track_to_id  = info["track_to_id"]

    if driver not in driver_to_id:
        raise ValueError(f"{driver} is not in the training set. "f"Trained drivers: {sorted(driver_to_id)}")
    if race_name not in track_to_id:
        raise ValueError(f"{race_name} is not in the training set. "f"Trained tracks: {sorted(track_to_id)}")

    laps = fetch_race(year, race_name)

    laps = laps.dropna(subset=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time","SpeedST", "TyreLife", "Compound", "AirTemp", "TrackTemp", "Rainfall",])
    laps = laps[(laps["LapTime"] >= minimum_lap_time) & (laps["LapTime"] <= max_lap_time)]
    laps = laps[laps["LapNumber"] > 1]
    laps["CompoundEncoded"] = laps["Compound"].map(compound_map).fillna(2).astype(int)

    laps = add_stint_features(laps)
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()].copy()

    laps["DriverID"] = laps["Driver"].map(driver_to_id).fillna(0).astype(int)
    laps["TrackID"]  = laps["Race"].map(track_to_id).fillna(0).astype(int)

    drv_laps = (laps[laps["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True))

    if len(drv_laps) < lookback_context:
        raise ValueError(f"Only {len(drv_laps)} clean laps available for {driver} — "f"need at least {lookback_context}.")

    windows, meta = [], []
    for i in range(len(drv_laps) - lookback_context + 1):
        raw = drv_laps[all_features].iloc[i:i + lookback_context].values.astype(np.float32)
        scaled = raw.copy()
        scaled[:, :numerical_features] = scaler.transform(raw[:, :numerical_features])
        windows.append(scaled)
        meta.append({"start_lap": int(drv_laps["LapNumber"].iloc[i + lookback_context - 1]) + 1, "last_lap_time": float(drv_laps["LapTime"].iloc[i + lookback_context - 1]),"compound": drv_laps["Compound"].iloc[i + lookback_context - 1],"stint":int(drv_laps["StintNumber"].iloc[i + lookback_context - 1]),})
    tensor = torch.tensor(np.array(windows, dtype=np.float32))
    return tensor, meta, drv_laps
