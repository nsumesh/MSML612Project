from src.data.fetch import fetch_all_races, RACES
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
    fetch_all_races(RACES)

    print("Data being preprocessed")
    run_preprocessing()

    print("Data being split")
    split()

    print("Complete.")
