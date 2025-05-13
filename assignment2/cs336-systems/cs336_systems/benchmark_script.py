import timeit
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from cs336_basics.Transformer import Transformer
from cs336_basics.Crossentropy import crossentropy
from cs336_basics.AdamW import AdamW
from tqdm.notebook import tqdm

def run_step(model, inputs, targets, optimizer, loss_fn, enable_backward=True):
    with record_function('forward_pass'):
        logits = model.forward(inputs)

    if enable_backward:
         with record_function('backward_pass'):
             loss = loss_fn(logits, targets); 
             loss.backward()
         with record_function('optimizer'):
             optimizer.step()
             optimizer.zero_grad(set_to_none=True)

def benchmark(w_steps=None, n_steps=5, d_model=768, d_ff=3072, num_layers=12, num_heads=12, use_profiler=False):
    total_times = []
    forward_times = []
    backward_times = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    context_length = 128
    vocab_size = 10000
    
    model = Transformer(vocab_size, 
                        context_length, 
                        d_model, 
                        num_layers, 
                        num_heads, 
                        d_ff, 
                        0.1, 
                        0.1).to(device)
    
    optim = AdamW(model.parameters())
    
    if w_steps is not None:
        for _ in tqdm(range(w_steps)):
            x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
            y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
            optim.zero_grad()
            logits = model(x)
            loss = crossentropy(logits, y)
            loss.backward()
            optim.step()

    if use_profiler:
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
            ) as prof:
                for _ in range(n_steps):
                    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
                    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
                    run_step(model, x, y, optim, crossentropy)
                    prof.step()

        prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        return
        

    for i in tqdm(range(n_steps)):
        optim.zero_grad()
        x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        start_time = timeit.default_timer()
        logits = model(x)
        forward_times.append(timeit.default_timer() - start_time)
        loss = crossentropy(logits, y)
        backward_start_time = timeit.default_timer()
        loss.backward()
        end_time = timeit.default_timer()
        optim.step()
        backward_times.append(end_time - backward_start_time)
        total_times.append(end_time - start_time)

    return torch.tensor(forward_times), torch.tensor(backward_times), torch.tensor(total_times)
    
                
    