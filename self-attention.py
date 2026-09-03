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
    sc = q@k.T
    tril = torch.tril(torch.ones(q.shape))
    mask = torch.masked_fill(tril==0, float('-inf'))
