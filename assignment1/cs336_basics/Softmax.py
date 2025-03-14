import torch

def softmax(x, dim=-1):
    max = torch.max(x, dim, keepdim=True).values
    x = x - max
    exp = torch.exp(x)
    sum = torch.sum(exp, dim=dim, keepdim=True)
    x = exp / sum
        
    return x