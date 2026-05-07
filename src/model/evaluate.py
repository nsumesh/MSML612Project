import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import numpy as np
import pandas as pd
from src.model.transformer import LapTimeTransformer
from src.data.dataset_file import get_dataloaders


def evaluate(model_path="models/best_model.pth", meta_path="data/splits/meta_test.csv", batch_size=64, horizon=5, model_kwargs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader, validation_loader, test_loader = get_dataloaders(batch_size=batch_size)
    meta = pd.read_csv(meta_path)

    info = json.loads(Path("data/splits/info.json").read_text())
    model = LapTimeTransformer(**{**info, **(model_kwargs or {})}).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, label in test_loader:
            output = model(data.to(device))
            preds = output[:, -horizon:]
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    meta = meta.iloc[:len(all_preds)].copy()
    last_lap = meta["last_lap_time"].values[:, None]  
    all_preds  = all_preds  + last_lap
    all_labels = all_labels + last_lap
    mae  = np.abs(all_preds - all_labels).mean()
    rmse = np.sqrt(((all_preds - all_labels) ** 2).mean())
    print("-" * 50)
    print(f"Overall Test MAE  : {mae:.3f}s")
    print(f"Overall Test RMSE : {rmse:.3f}s")
    print("\nMAE per predicted lap:")
    for i in range(horizon):
        print(f"Lap +{i+1}: {np.abs(all_preds[:, i] - all_labels[:, i]).mean():.3f}s")
    meta["mae"] = np.abs(all_preds - all_labels).mean(axis=1)
    print("\n" + "-" * 50)
    print("MAE by Race:")
    race_summary = (meta.groupby(["year", "race"])["mae"].agg(["mean", "count"]).rename(columns={"mean": "MAE (s)", "count": "windows"}).sort_values("MAE (s)"))
    print(race_summary.to_string())
    print("\n" + "-" * 50)
    print("MAE by Driver (top 10 best):")
    driver_summary = (meta.groupby("driver")["mae"].agg(["mean", "count"]).rename(columns={"mean": "MAE (s)", "count": "windows"}).sort_values("MAE (s)").head(10))
    print(driver_summary.to_string())
    print("\n" + "-" * 50)
    print("MAE by Driver (top 10 worst):")
    print(meta.groupby("driver")["mae"].mean().sort_values(ascending=False).head(10).to_string())
    return all_preds, all_labels, meta

if __name__ == "__main__":
    evaluate()
