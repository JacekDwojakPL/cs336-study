import torch.nn as nn
from cs336_basics.MultiHeadAttention import MultiHeadAttention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.FeedForward import FeedForward

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop):
        super(Block, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.drop1 = nn.Dropout(residual_pdrop)
        
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.drop2 = nn.Dropout(residual_pdrop)
        
    def forward(self, x):
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        
        return x
    
    def load_state_dict(self, state_dict):
        self.ln1.weight.data = state_dict["ln1.weight"]
        self.attn.k_proj.weight.data = state_dict["attn.k_proj.weight"]
        self.attn.q_proj.weight.data = state_dict["attn.q_proj.weight"]
        self.attn.v_proj.weight.data = state_dict["attn.v_proj.weight"]
        self.attn.output_proj.weight.data = state_dict["attn.output_proj.weight"]
        self.ln2.weight.data = state_dict["ln2.weight"]
        self.ffn.w1.weight.data = state_dict["ffn.w1.weight"]
        self.ffn.w2.weight.data = state_dict["ffn.w2.weight"]