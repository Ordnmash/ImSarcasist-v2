import torch
import torch.nn as nn

n_embd = 10
block_size = 10

inputV = torch.randn((block_size, n_embd), dtype=torch.float32)

query = nn.Linear(n_embd, n_embd, bias=False)
key   = nn.Linear(n_embd, n_embd, bias=False)
value = nn.Linear(n_embd, n_embd, bias=False)

q = query(inputV)
k = key(inputV)
v = value(inputV)

wei = q @ k.T
attention = wei @ v
