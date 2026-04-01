import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class F1LapDataset(Dataset):

    def __init__(self, data_dir: str ="data/splits/"):
        data_dir = Path(data_dir)
        numpy_data_path = data_dir/"data.npy"
        labels_data_path = data_dir/"labels.npy"
        self.data = torch.tensor(np.load(numpy_data_path), dtype = torch.float32)
        self.labels = torch.tensor(np.load(labels_data_path), dtype = torch.float32) 
        