import torch
import torch.nn as nn


class Head(nn.Module):

    def __init__(self):
        super(Head, self).__init__()

    def forward(self, x):
        return x
