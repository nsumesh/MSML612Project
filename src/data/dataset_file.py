"""
PyTorch Dataset and DataLoader setup for the preprocessed split files.

F1LapDataset just loads data_{split}.npy and labels_{split}.npy from disk and wraps
them as float32 tensors. get_dataloaders builds all three splits at once and returns
their DataLoaders — training data is shuffled, val and test are not. That's really
all there is here; the heavy lifting all happens in preprocess.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class F1LapDataset(Dataset):

    def __init__(self, split: str, data_dir: str = "data/splits/"):
        data_dir = Path(data_dir)
        self.data = torch.tensor(np.load(data_dir / f"data_{split}.npy"), dtype=torch.float32)
        self.labels = torch.tensor(np.load(data_dir / f"labels_{split}.npy"), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def get_dataloaders(data_dir: str = "data/splits/", batch_size: int = 64):
    training_dataset = F1LapDataset("train", data_dir)
    validation_dataset = F1LapDataset("val",   data_dir)
    test_dataset = F1LapDataset("test",  data_dir)

    training_dataloader = DataLoader(training_dataset,batch_size=batch_size, shuffle=True,  num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    testing_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=0)
    return training_dataloader, validation_dataloader, testing_dataloader

if __name__ == "__main__":
    training_dataloader, validation_dataloader, testing_dataloader = get_dataloaders(batch_size=64)
