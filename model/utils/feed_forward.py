import torch.nn as nn
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_in, d_ff, d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
