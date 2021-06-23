import torch
import torch.nn as nn


class DSConvBlock(nn.Module):

    def __init__(self, in_channels):
        super(DSConvBlock, self).__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), padding=1, groups=in_channels)
        self.point_wise = nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), padding=0)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.9997, eps=4e-5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiFPN(nn.Module):

    """
        P7_0 -------------------------- P7_2 -------->
        P6_0 ---------- P6_1 ---------- P6_2 -------->
        P5_0 ---------- P5_1 ---------- P5_2 -------->
        P4_0 ---------- P4_1 ---------- P4_2 -------->
        P3_0 -------------------------- P3_2 -------->
    """

    def __init__(self, in_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()

        # common
        self.epsilon = epsilon
        self.relu = nn.ReLU()

        # resize
        self.upsample = nn.Upsample(scale_factor=2)
        self.dnsample = nn.MaxPool2d(kernel_size=2)

        # conv layers
        self.conv_p6_1 = DSConvBlock(in_channels)
        self.conv_p5_1 = DSConvBlock(in_channels)
        self.conv_p4_1 = DSConvBlock(in_channels)
        self.conv_p7_2 = DSConvBlock(in_channels)
        self.conv_p6_2 = DSConvBlock(in_channels)
        self.conv_p5_2 = DSConvBlock(in_channels)
        self.conv_p4_2 = DSConvBlock(in_channels)
        self.conv_p3_2 = DSConvBlock(in_channels)

        # weights
        self.w_p6_1 = nn.Parameter(torch.ones(2))
        self.w_p5_1 = nn.Parameter(torch.ones(2))
        self.w_p4_1 = nn.Parameter(torch.ones(2))
        self.w_p7_2 = nn.Parameter(torch.ones(2))
        self.w_p6_2 = nn.Parameter(torch.ones(3))
        self.w_p5_2 = nn.Parameter(torch.ones(3))
        self.w_p4_2 = nn.Parameter(torch.ones(3))
        self.w_p3_2 = nn.Parameter(torch.ones(2))

    def forward(self, x):

        p3_0, p4_0, p5_0, p6_0, p7_0 = x

        w_p6_1 = self.relu(self.w_p6_1)
        w_p5_1 = self.relu(self.w_p5_1)
        w_p4_1 = self.relu(self.w_p4_1)
        w_p7_2 = self.relu(self.w_p7_2)
        w_p6_2 = self.relu(self.w_p6_2)
        w_p5_2 = self.relu(self.w_p5_2)
        w_p4_2 = self.relu(self.w_p4_2)
        w_p3_2 = self.relu(self.w_p3_2)

        w_p6_1 = w_p6_1 / (torch.sum(w_p6_1, dim=0) + self.epsilon)
        w_p5_1 = w_p5_1 / (torch.sum(w_p5_1, dim=0) + self.epsilon)
        w_p4_1 = w_p4_1 / (torch.sum(w_p4_1, dim=0) + self.epsilon)
        w_p7_2 = w_p7_2 / (torch.sum(w_p7_2, dim=0) + self.epsilon)
        w_p6_2 = w_p6_2 / (torch.sum(w_p6_2, dim=0) + self.epsilon)
        w_p5_2 = w_p5_2 / (torch.sum(w_p5_2, dim=0) + self.epsilon)
        w_p4_2 = w_p4_2 / (torch.sum(w_p4_2, dim=0) + self.epsilon)
        w_p3_2 = w_p3_2 / (torch.sum(w_p3_2, dim=0) + self.epsilon)

        p6_1 = self.conv_p6_1(w_p6_1[0] * p6_0 + w_p6_1[1] * self.upsample(p7_0))
        p5_1 = self.conv_p5_1(w_p5_1[0] * p5_0 + w_p5_1[1] * self.upsample(p6_1))
        p4_1 = self.conv_p4_1(w_p4_1[0] * p4_0 + w_p4_1[1] * self.upsample(p5_1))
        p3_2 = self.conv_p3_2(w_p3_2[0] * p3_0 + w_p3_2[1] * self.upsample(p4_1))
        p4_2 = self.conv_p4_2(w_p4_2[0] * p4_0 + w_p4_2[1] * p4_1 + w_p4_2[2] * self.dnsample(p3_2))
        p5_2 = self.conv_p5_2(w_p5_2[0] * p5_0 + w_p5_2[1] * p5_1 + w_p5_2[2] * self.dnsample(p4_2))
        p6_2 = self.conv_p6_2(w_p6_2[0] * p6_0 + w_p6_2[1] * p6_1 + w_p6_2[2] * self.dnsample(p5_2))
        p7_2 = self.conv_p7_2(w_p7_2[0] * p7_0 + w_p7_2[1] * self.dnsample(p6_2))

        return p3_2, p4_2, p5_2, p6_2, p7_2
