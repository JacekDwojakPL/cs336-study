import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        super(AdamW, self).__init__(params, {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps})
         
    def step(self, closure = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)
                g = p.grad
                m = (b1*m) + (1-b1)*g
                v = (b2*v) + (1-b2)*g**2
                at = lr * (((1-b2**t)**0.5) / (1-b1**t))
                p.data -= at * (m / (v**0.5 + eps))
                p.data -= lr*weight_decay*p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t+1
                        
        return loss
    
    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr