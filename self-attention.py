import torch
import torch.nn as nn

query = nn.Linear(20, 10, bias=False)
key   = nn.Linear(20, 10, bias=False)
value = nn.Linear(20, 10, bias=False)
