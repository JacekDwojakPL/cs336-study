import torch

def gelu(x):
    erf = 0.5 * (1 + torch.erf_(x / torch.sqrt(torch.tensor(2))))
    x = x*erf
    
    return x