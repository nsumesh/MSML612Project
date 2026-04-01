import pandas as pd
import numpy as np
from pathlib import Path

def check_raw():
    path = "data/raw/all_laps.csv"
    if not Path(path).exists():
        print("MISSING: data/raw/all_laps.csv")
        return
    df = pd.read_csv(path)
    print(f"[raw] shape: {df.shape}")
    print(f"[raw] races: {sorted(df['Race'].unique())}")
    print(f"[raw] years: {sorted(df['Year'].unique())}")
    print(f"[raw] null counts:\n{df.isnull().sum()}")
    print(f"[raw] LapTime range: {df['LapTime'].min():.2f} - {df['LapTime'].max():.2f}")

def check_processed():
    data_path = "data/processed/data.npy"
    labels_path = "data/processed/labels.npy"
    if not Path(data_path).exists() or not Path(labels_path).exists():
        print("MISSING: data/processed/")
        return
    data = np.load(data_path)
    labels = np.load(labels_path)
    print(f"[processed] data shape: {data.shape}  (samples, lookback, features)")
    print(f"[processed] labels shape: {labels.shape}  (samples, horizon)")
    print(f"[processed] data NaNs: {np.isnan(data).sum()}")
    print(f"[processed] labels NaNs: {np.isnan(labels).sum()}")
    print(f"[processed] data mean: {data.mean():.4f}, std: {data.std():.4f}")

def check_splits():
    splits = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    for name in splits:
        path = f"data/splits/{name}.npy"
        if not Path(path).exists():
            print(f"MISSING: {path}")
            continue
        arr = np.load(path)
        print(f"[splits] {name}: {arr.shape}")

if __name__ == "__main__":
    print("=== RAW ===")
    check_raw()
    print("\n=== PROCESSED ===")
    check_processed()
    print("\n=== SPLITS ===")
    check_splits()
