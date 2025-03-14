import torch.nn as nn
from cs336_basics.Gelu import gelu


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        x = self.w1(x)
        x = gelu(x)
        x = self.w2(x)
    
        return x
    
    def load_state_dict(self, state_dict):
        self.w1.weight.data = state_dict["w1.weight"]
        self.w2.weight.data = state_dict["w2.weight"]
         