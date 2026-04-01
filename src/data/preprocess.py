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


def build_transformer_sequences(laps, lookback = 30, horizon = 10):
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
        
    data, target_labels, metadata = [],[],[]
    for (race, year, driver),group in laps.groupby(["Race", "Year", "Driver"]):
        group = group.sort_values("LapNumber").reset_index(drop=True)
        if len(group)<lookback+horizon:
            continue

        for i in range(len(group) - lookback - horizon+1):
            data_window = group[features].iloc[i:i+lookback].values
            label_window = group[target].iloc[i+lookback:i+lookback+horizon].values
            data.append(data_window)
            target_labels.append(label_window)
            metadata.append({"race":race, "year":year, "driver": driver, "start_lap" : i})
    
    return np.array(data, dtype = np.float32), np.array(target_labels, dtype=np.float32), metadata

def run_preprocessing(csv_path="data/raw/all_laps.csv"):
    laps = pd.read_csv(csv_path)
    laps = clean_laps(laps)
    data, labels, metadata = build_transformer_sequences(laps)
    return normalize_and_save(data, labels)

def normalize_and_save(data, labels, save_dir = "data/processed/"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    n, seq, feat = data.shape
    scaler = StandardScaler()
    data_flattened = data.reshape(-1, feat)
    data_scaled = scaler.fit_transform(data_flattened).reshape(n, seq, feat)
    np.save(f"{save_dir}/data.npy", data_scaled)
    np.save(f"{save_dir}/labels.npy", labels)
    with open(f"{save_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler,f)
    return data_scaled, labels, scaler
