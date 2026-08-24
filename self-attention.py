import torch
import torch.nn as nn

n_embd = 10
block_size = 10

query = nn.Linear(n_embd, n_embd, bias=False)
key   = nn.Linear(n_embd, n_embd, bias=False)
value = nn.Linear(n_embd, n_embd, bias=False)
