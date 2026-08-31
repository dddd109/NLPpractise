import torch
import torch.nn as nn

class GeLUFFN(nn.Module):
    def __init__(self,dim,hidden_dim = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4*dim
        self.c_fc = nn.Linear(dim,hidden_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(hidden_dim,dim)

        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self,x):
        x = self.gelu(self.c_fc(x))
        return self.c_proj(x)