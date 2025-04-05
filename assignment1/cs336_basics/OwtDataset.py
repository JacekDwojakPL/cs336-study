import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class OwtDataset(Dataset):
    def __init__(self, data, context_length, device):
        super(OwtDataset, self).__init__()
        self.data = data
        self.device = device
        self.context_length = context_length
    
    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, index):
        x = self.data[index: index + self.context_length].astype(np.int64)
        y = self.data[index+1:index+1+self.context_length].astype(np.int64)
        
        return (torch.tensor(x).to(self.device), torch.tensor(y).to(self.device))
    
    
def get_batch(dataset, batch_size, context_length, device):
    dataset = OwtDataset(dataset, context_length, device)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    
    return next(iter(dataloader))