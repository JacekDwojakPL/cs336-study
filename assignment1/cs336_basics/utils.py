import torch
import numpy as np


def save_checkpoint(model, optimizer, iteration, out):
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "iteration": iteration}, out)
    
def load_checkpoint(src, model, optimizer):
    state = torch.load(src)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    
    return state["iteration"]

def lr_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return (it / warmup_iters)*max_learning_rate
    if it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + np.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * np.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
    
def gradient_clipping(parameters, max_norm):
    norm = sum([torch.sum(p.grad**2) for p in filter(lambda p: p.grad != None, parameters)])
    norm = norm ** 0.5
    
    if norm > max_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.detach().mul_(max_norm / (norm+1e-6))