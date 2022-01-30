import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size_in, size_out, dropout,for_input_sublayer=False):
        super(SublayerConnection, self).__init__()
        self.for_input_sublayer = for_input_sublayer
        self.norm = LayerNorm(size_in)
        self.dropout = nn.Dropout(dropout)
        # self.use_shoutcut_Liner = False
        # if size_in == size_out:
        #     self.use_shoutcut_Liner = True
        #     self.down = nn.Linear(in_features=size_in, out_features=size_out)

    def forward(self, x, sublayer,A=None):
        "Apply residual connection to any sublayer with the same size."
        if self.for_input_sublayer:
            x_output = self.dropout(sublayer(self.norm(x),A))
        else:
            x_output = self.dropout(sublayer(self.norm(x)))
        # if self.use_shoutcut_Liner:
        #     x_output = self.down(x) + x_output
        # else:
        #     x_output = x + x_output
        if x.size(-1) == x_output.size(-1):
            x_output = x_output + x
        return x_output
