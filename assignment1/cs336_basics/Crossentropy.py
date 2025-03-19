import torch

def crossentropy(logits, targets):
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -torch.sum(logits[torch.arange(len(targets)), targets]) / len(targets)
    
    return loss