import torch
import torch.nn as nn
from cs336_basics.Block import Block
from cs336_basics.Softmax import softmax
from cs336_basics.RMSNorm import RMSNorm
from collections import OrderedDict

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, attn_prdop, residual_pdrop):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdorp = attn_prdop
        self.residual_pdrop = residual_pdrop
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.Sequential(*[Block(d_model, num_heads, d_ff, attn_prdop, residual_pdrop) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, x):
        _, T = x.size()
        token_embeddings = self.token_embeddings(x)
        position_embeddings = self.position_embeddings(torch.arange(T, device=self.device))
        embeddings = token_embeddings + position_embeddings
        embeddings = nn.functional.dropout(embeddings, self.residual_pdrop)
        x = self.layers(embeddings)
        x = self.ln_final(x)
        x = self.lm_head(x)
        
        return x
    
    def load_state_dict(self, state_dict):
        self.token_embeddings.weight.data = state_dict["token_embeddings.weight"]
        self.position_embeddings.weight.data = state_dict["position_embeddings.weight"]
        for i in range(self.num_layers):
            weights = OrderedDict()
            weights["ln1.weight"] = state_dict[f"layers.{i}.ln1.weight"]
            weights["attn.k_proj.weight"] = state_dict[f"layers.{i}.attn.k_proj.weight"]
            weights["attn.q_proj.weight"] = state_dict[f"layers.{i}.attn.q_proj.weight"]
            weights["attn.v_proj.weight"] = state_dict[f"layers.{i}.attn.v_proj.weight"]
            weights["attn.output_proj.weight"] = state_dict[f"layers.{i}.attn.output_proj.weight"]
            weights["ln2.weight"] = state_dict[f"layers.{i}.ln2.weight"]
            weights["ffn.w1.weight"] = state_dict[f"layers.{i}.ffn.w1.weight"]
            weights["ffn.w2.weight"] = state_dict[f"layers.{i}.ffn.w2.weight"]
            self.layers[i].load_state_dict(weights)
        self.ln_final.weight.data = state_dict["ln_final.weight"]
        self.lm_head.weight.data = state_dict["lm_head.weight"]
