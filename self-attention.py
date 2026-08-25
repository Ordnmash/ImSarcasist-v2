import torch
import torch.nn as nn
import torch.nn.functional as F

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
wei = wei / torch.sqrt(torch.tensor(k.shape[0])) # normalizing before softmax
attention = F.softmax(wei, dim=0) @ v            # note there's no masked fill in this example
