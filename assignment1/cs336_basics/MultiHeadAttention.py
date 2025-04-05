import torch
import torch.nn as nn
from cs336_basics.Attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.pdrop = attn_pdrop
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
    def forward(self, input_features):
        B, T, C = input_features.shape
        k = self.k_proj(input_features).view(B,T,self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(input_features).view(B,T,self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(input_features).view(B,T,self.num_heads, self.head_dim).transpose(1, 2)
        mask = torch.triu(torch.ones((T, T)), diagonal=1).bool()
        attention = scaled_dot_product_attention(k, q, v, mask=mask, pdrop=self.pdrop)
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)
        ret = self.output_proj(attention)

        return ret
    
    def load_state_dict(self, state_dict):
        self.k_proj.weight.data = torch.cat([state_dict[f"k_heads.{i}.weight"]for i in range(self.num_heads)])
        self.q_proj.weight.data = torch.cat([state_dict[f"q_heads.{i}.weight"]for i in range(self.num_heads)])
        self.v_proj.weight.data = torch.cat([state_dict[f"v_heads.{i}.weight"]for i in range(self.num_heads)])
        self.output_proj.weight.data = state_dict["output_proj.weight"]