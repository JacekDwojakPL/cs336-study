import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        mean = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        x = x / mean
        x = x*self.weight
        return x
    
    def load_state_dict(self, state_dict):
        self.weight.data = state_dict["weight"]