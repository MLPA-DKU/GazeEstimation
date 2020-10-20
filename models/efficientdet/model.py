import torch
import torch.nn as nn
import efficientnet_pytorch as ef


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


class BiFPN(nn.Module):

    def __init__(self, in_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.relu = nn.ReLU()

        # conv layers
        self.conv3_up = DSConvBlock(in_channels)
        self.conv4_up = DSConvBlock(in_channels)
        self.conv5_up = DSConvBlock(in_channels)
        self.conv6_up = DSConvBlock(in_channels)
        self.conv4_dn = DSConvBlock(in_channels)
        self.conv5_dn = DSConvBlock(in_channels)
        self.conv6_dn = DSConvBlock(in_channels)
        self.conv7_dn = DSConvBlock(in_channels)

        # feature scaling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dnsample = nn.MaxPool2d(kernel_size=2)

        # weights
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p6_w1 = nn.Parameter(torch.ones(2))

        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2 = nn.Parameter(torch.ones(3))
        self.p7_w2 = nn.Parameter(torch.ones(2))

    def forward(self, x):
        """
            P7_0 -------------------------- P7_2 -------->
            P6_0 ---------- P6_1 ---------- P6_2 -------->
            P5_0 ---------- P5_1 ---------- P5_2 -------->
            P4_0 ---------- P4_1 ---------- P4_2 -------->
            P3_0 -------------------------- P3_2 -------->
        """

        p3_0, p4_0, p5_0, p6_0, p7_0 = x

        p3_w1 = self.relu(self.p3_w1)
        p4_w1 = self.relu(self.p4_w1)
        p5_w1 = self.relu(self.p5_w1)
        p6_w1 = self.relu(self.p6_w1)
        p4_w2 = self.relu(self.p4_w2)
        p5_w2 = self.relu(self.p5_w2)
        p6_w2 = self.relu(self.p6_w2)
        p7_w2 = self.relu(self.p7_w2)

        p3_w1 = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p4_w1 = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p5_w1 = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p6_w1 = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p4_w2 = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p5_w2 = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p6_w2 = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p7_w2 = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)

        p6_1 = self.conv6_up(p6_w1[0] * p6_0 + p6_w1[1] * self.upsample(p7_0))
        p5_1 = self.conv5_up(p5_w1[0] * p5_0 + p5_w1[1] * self.upsample(p6_1))
        p4_1 = self.conv4_up(p4_w1[0] * p4_0 + p4_w1[1] * self.upsample(p5_1))

        p3_2 = self.conv3_up(p3_w1[0] * p3_0 + p3_w1[1] * self.upsample(p4_1))
        p4_2 = self.conv4_dn(p4_w2[0] * p4_0 + p4_w2[1] * p4_1 + p4_w2[2] * self.dnsample(p3_2))
        p5_2 = self.conv5_dn(p5_w2[0] * p5_0 + p5_w2[1] * p5_1 + p4_w2[2] * self.dnsample(p4_2))
        p6_2 = self.conv6_dn(p6_w2[0] * p6_0 + p6_w2[1] * p6_1 + p4_w2[2] * self.dnsample(p5_2))
        p7_2 = self.conv7_dn(p7_w2[0] * p7_0 + p7_w2[1] * self.dnsample(p6_2))

        return p3_2, p4_2, p5_2, p6_2, p7_2


class EfficientNet(nn.Module):

    def __init__(self):
        super(EfficientNet, self).__init__()
        model = ef.EfficientNet.from_pretrained('efficientnet-b0')
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        feature_maps = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(x)
        return feature_maps[1:]


class EfficientDet(nn.Module):

    def __init__(self, compound_coef=0):
        super(EfficientDet, self).__init__()
        self.compound_coef = compound_coef
        self.out_channels = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]
        self.backbone = EfficientNet()

        self.conv3 = nn.Conv2d(40, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(80, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(192, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(192, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
        )

        self.bifpn = nn.Sequential(*[BiFPN(self.out_channels) for _ in range(min(2 + self.compound_coef, 8))])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.upsample = nn.Upsample((64, 64), mode='nearest')

        self.regression = nn.Sequential(
            nn.Linear(320, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)

        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)

        features = p3, p4, p5, p6, p7
        features = self.bifpn(features)
        features_for_regression = torch.flatten(torch.cat([self.gap(f) for f in features], dim=1), 1)
        # features_for_alignments = torch.flatten(torch.cat([self.upsample(f) for f in features], dim=1), 1)

        estimated_gaze = self.regression(features_for_regression)

        return estimated_gaze
