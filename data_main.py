"""
Data Pipeline script that kicks off the full data pipeline from scratch.

Run this once before training. It first downloads every race from the 2024 and 2025
seasons via FastF1 and saves everything to a single CSV at data/raw/all_laps.csv.
Then it hands that CSV off to the preprocessing pipeline, which cleans the laps,
builds sliding windows, fits the scaler, and saves the train/val/test splits under
data/splits/ — ready for the model to pick up.
"""

from src.data.fetch import fetch_all_races, races
from src.data.preprocess import run_preprocessing
import numpy as np
from pathlib import Path

def split(data_path="data/processed/data.npy", labels_path="data/processed/labels.npy", save_dir="data/splits/"):
    data = np.load(data_path)
    labels = np.load(labels_path)
    n = len(data)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_dir}/data_train.npy", data[:train_end])
    np.save(f"{save_dir}/labels_train.npy", labels[:train_end])
    np.save(f"{save_dir}/data_val.npy",   data[train_end:val_end])
    np.save(f"{save_dir}/labels_val.npy",   labels[train_end:val_end])
    np.save(f"{save_dir}/data_test.npy",  data[val_end:])
    np.save(f"{save_dir}/labels_test.npy",  labels[val_end:])

if __name__ == "__main__":
    print("Races being fetched")
    fetch_all_races(races)
    print("Data being preprocessed")
    run_preprocessing()
    print("Complete.")
