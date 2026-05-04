"""
Hyperparameter sweep for the lap-time transformer.

Each config is trained, then evaluated on the test set. Results are appended to
`results/sweep.csv` so you can re-run / extend without losing history.

Usage:
    python sweep.py
"""
import csv
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset_file import get_dataloaders
from src.model.transformer import LapTimeTransformer
from src.model.training import train_transformer


SWEEP_CSV = Path("results/sweep.csv")
HORIZON = 5


def evaluate_test_mae(ckpt_path, batch_size, model_kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=batch_size)
    model = LapTimeTransformer(**(model_kwargs or {})).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            out = model(data)
            preds.append(out[:, -HORIZON:].cpu().numpy())
            labels.append(label.numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return float(np.abs(preds - labels).mean())


def append_result(row):
    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SWEEP_CSV.exists()
    with open(SWEEP_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_config(name, train_kwargs, model_kwargs):
    print(f"\n{'=' * 70}\nRunning: {name}\n{'=' * 70}")
    t0 = time.time()
    try:
        result = train_transformer(
            run_name=name,
            model_kwargs=model_kwargs,
            verbose=True,
            **train_kwargs,
        )
        test_mae = evaluate_test_mae(
            ckpt_path=result["ckpt_path"],
            batch_size=train_kwargs.get("batch_size", 64),
            model_kwargs=model_kwargs,
        )
        elapsed = time.time() - t0
        row = {
            "run_name": name,
            "test_mae": round(test_mae, 4),
            "best_val_mae": round(result["best_val_mae"], 4),
            "epochs_run": result["epochs_run"],
            "elapsed_sec": round(elapsed, 1),
            **{f"train_{k}": v for k, v in train_kwargs.items()},
            **{f"model_{k}": v for k, v in (model_kwargs or {}).items()},
        }
        append_result(row)
        print(f"\n[{name}] test MAE = {test_mae:.4f} ({elapsed:.0f}s)")
    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# Configs to sweep. Edit this list to add/remove runs.
# -----------------------------------------------------------------------------
CONFIGS = [
    # Phase 1: training-loop changes (MSE, scheduler+ES, longer)
    ("p1_mse_baseline", dict(epochs=150, lr=1e-3, batch_size=64, weight_decay=1e-4, loss_name="mse"), None),
    ("p1_mse_lr3e4",    dict(epochs=150, lr=3e-4, batch_size=64, weight_decay=1e-4, loss_name="mse"), None),
    ("p1_mse_bs128",    dict(epochs=150, lr=1e-3, batch_size=128, weight_decay=1e-4, loss_name="mse"), None),
    ("p1_mse_wd1e3",    dict(epochs=150, lr=1e-3, batch_size=64, weight_decay=1e-3, loss_name="mse"), None),

    # Phase 2: architecture (more capacity since model is under-fitting)
    ("p2_d128_l3",      dict(epochs=150, lr=5e-4, batch_size=64, weight_decay=1e-4, loss_name="mse"),
                        dict(d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1)),
    ("p2_d128_l4_h8",   dict(epochs=150, lr=5e-4, batch_size=64, weight_decay=1e-4, loss_name="mse"),
                        dict(d_model=128, n_heads=8, n_layers=4, d_ff=512, dropout=0.1)),
    ("p2_d192_l3_h8",   dict(epochs=150, lr=3e-4, batch_size=64, weight_decay=1e-4, loss_name="mse"),
                        dict(d_model=192, n_heads=8, n_layers=3, d_ff=384, dropout=0.15)),
    ("p2_d128_l4_drop2",dict(epochs=150, lr=5e-4, batch_size=64, weight_decay=1e-4, loss_name="mse"),
                        dict(d_model=128, n_heads=8, n_layers=4, d_ff=512, dropout=0.2)),
    ("p2_d256_l3_h8",   dict(epochs=150, lr=3e-4, batch_size=64, weight_decay=1e-4, loss_name="mse"),
                        dict(d_model=256, n_heads=8, n_layers=3, d_ff=512, dropout=0.15)),
]


def main():
    print(f"Sweeping {len(CONFIGS)} configs. Logging to {SWEEP_CSV}")
    for name, train_kwargs, model_kwargs in CONFIGS:
        run_config(name, train_kwargs, model_kwargs)

    if SWEEP_CSV.exists():
        df = pd.read_csv(SWEEP_CSV).sort_values("test_mae")
        print("\n" + "=" * 70)
        print("SWEEP RESULTS (sorted by test_mae)")
        print("=" * 70)
        print(df[["run_name", "test_mae", "best_val_mae", "epochs_run", "elapsed_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
