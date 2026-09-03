import torch
import torch.nn as nn
import torch.nn.functional as F

# let's build SelfAttention block from scratch
class SelfAttention(nn.Module):
  def __init__(self, insize, outsize):
    super().__init__()
    self.query = nn.Linear(insize, outsize, bias=False)
    self.key   = nn.Linear(insize, outsize, bias=False)
    self.value = nn.Linear(insize, outsize, bias=False)

  def forward(self, x):
    q = self.query(x)
    v = self.value(x)
    k = self.key(x)
    # compute the attention scores
    sc = (q@k.T) / (k.shape[-1]**0.5) # normalize scores before softmax
    tril = torch.tril(torch.ones(q.shape))
    wei  = sc.masked_fill(tril==0, float('-inf'))
    wei  = wei.softmax(dim=-1)
    att  = wei@v
    return att # Attention(Q,K,V) = Softmax(qkT/dk)V
