'''
Portions of this code copyright 2017, Clement Pinard
'''

import torch
import torch.nn as nn
import math


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target, p=2, dim=1).mean()
        return lossvalue

