import torch.nn as nn


class DSConvBlock(nn.Module):

    def __init__(self, in_channels):
        super(DSConvBlock, self).__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.point_wise = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.9997, eps=4e-5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
