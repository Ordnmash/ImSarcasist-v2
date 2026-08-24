import torch
import torch.nn as nn

q = nn.Linear(20, 10, bias=False)
k = nn.Linear(20, 10, bias=False)
v = nn.Linear(20, 10, bias=False)
