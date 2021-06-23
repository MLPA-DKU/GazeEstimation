import torch.nn as nn


class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, (1, 1)),  # equal with nn.Linear
            nn.ReLU()
        )
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels // reduction, in_channels, (1, 1)),  # equal with nn.Linear
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.squeeze(x)
        w = self.excitation(w)
        return w * x
