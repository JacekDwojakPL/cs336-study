import torch
from cs336_basics.Softmax import softmax

def scaled_dot_product_attention(K, Q, V, mask=None, pdrop=None):
    dim = K.shape[-1]
    attention_scores= Q @ K.transpose(-1, -2)
    attention_scores = attention_scores / torch.sqrt(torch.tensor(dim))
    
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask, -torch.inf)
    
    attention_weights = softmax(attention_scores)
    ret = attention_weights @ V
    
    if pdrop is not None:
        ret = torch.nn.functional.dropout(ret, pdrop)
    
    return ret