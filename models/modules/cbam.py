import torch
import torch.nn as nn


class ChannelGate(nn.Module):

    pooling = {
        'avg': nn.AdaptiveAvgPool2d(1),
        'max': nn.AdaptiveMaxPool2d(1),
    }

    def __init__(self, in_planes, reduction_ratio=16, pool_types=None):
        super(ChannelGate, self).__init__()
        self.in_planes = in_planes
        self.pool_types = pool_types if pool_types is not None else ['avg', 'max']
        self.channeling = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_planes // reduction_ratio, in_planes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_attr_sum = None
        for pool_type in self.pool_types:
            channel_attr_raw = self.channeling(torch.flatten(self.pooling[pool_type](x), 1))
            channel_attr_sum = channel_attr_raw if channel_attr_sum is None else channel_attr_sum + channel_attr_raw
        scale = self.sigmoid(channel_attr_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        compressed = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        features = self.conv(compressed)
        scale = self.sigmoid(features)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, in_planes, reduction_ratio, pool_types=None):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelGate(in_planes, reduction_ratio, pool_types)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x
