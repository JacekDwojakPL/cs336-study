import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class OwtDataset(Dataset):
    def __init__(self, dataset, context_length, device):
        super(OwtDataset, self).__init__()
        self.data = dataset
        self.context_length = context_length
        self.device = device

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.context_length].astype(np.int64), device=self.device)
        y = torch.tensor(self.data[idx+1:idx+1+self.context_length].astype(np.int64), device=self.device)

        return (x, y)

    def get_batch(self, batch_size):
        indices = torch.randint(0, len(self), (batch_size,), device=self.device)
        x = torch.stack([self[i][0].to(self.device) for i in indices])
        y = torch.stack([self[i][1].to(self.device) for i in indices])

        return (x, y)