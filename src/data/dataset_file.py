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
        
    def length(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index], self.labels[index]

def get_dataloaders(data_dir : str = "data/splits/", batch_size : int = 64):
    training_dataset = F1LapDataset("train", data_dir)
    validation_dataset = F1LapDataset("validation", data_dir)
    test_dataset = F1LapDataset("test", data_dir)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testing_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return training_dataloader, validation_dataloader, testing_dataloader

if __name__=="__main__":
    training_dataloader, validation_dataloader, testing_dataloader = get_dataloaders(batch_size=64)
    