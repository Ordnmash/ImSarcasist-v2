import torch
import torch.nn as nn
import torch.nn.functional as F

# let's build SelfAttention block from scratch
class SelfAttention(nn.Module):
  def __init__(self, insize, outsize):
    super().__init__()
    # attributes of SelfAttention follows
