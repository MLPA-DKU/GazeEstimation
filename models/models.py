import torch
import torch.nn as nn
import torchvision.models as models

import bottleneck_transformer_pytorch as bot


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
        self.conv1_up = DSConvBlock(in_channels)
        self.conv2_up = DSConvBlock(in_channels)
        self.conv3_up = DSConvBlock(in_channels)
        self.conv2_dn = DSConvBlock(in_channels)
        self.conv3_dn = DSConvBlock(in_channels)
        self.conv4_dn = DSConvBlock(in_channels)

        # feature scaling layers
        self.upsample = nn.Upsample(scale_factor=2)
        self.dnsample = nn.MaxPool2d(kernel_size=2)

        # weights
        self.p1_w1 = nn.Parameter(torch.ones(2))
        self.p2_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p2_w2 = nn.Parameter(torch.ones(3))
        self.p3_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2 = nn.Parameter(torch.ones(2))

    def forward(self, x):
        """
            P4_0 -------------------------- P4_2 -------->
            P3_0 ---------- P3_1 ---------- P3_2 -------->
            P2_0 ---------- P2_1 ---------- P2_2 -------->
            P1_0 -------------------------- P1_2 -------->
        """

        p1_0, p2_0, p3_0, p4_0 = x

        p1_w1 = self.relu(self.p1_w1)
        p2_w1 = self.relu(self.p2_w1)
        p3_w1 = self.relu(self.p3_w1)
        p2_w2 = self.relu(self.p2_w2)
        p3_w2 = self.relu(self.p3_w2)
        p4_w2 = self.relu(self.p4_w2)

        p1_w1 = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        p2_w1 = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p3_w1 = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p2_w2 = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p3_w2 = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p4_w2 = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)

        p3_1 = self.conv3_up(p3_w1[0] * p3_0 + p3_w1[1] * self.upsample(p4_0))
        p2_1 = self.conv2_up(p2_w1[0] * p2_0 + p2_w1[1] * self.upsample(p3_0))

        p1_2 = self.conv1_up(p1_w1[0] * p1_0 + p1_w1[1] * self.upsample(p2_1))
        p2_2 = self.conv2_dn(p2_w2[0] * p2_0 + p2_w2[1] * p2_1 + p2_w2[2] * self.dnsample(p1_2))
        p3_2 = self.conv3_dn(p3_w2[0] * p3_0 + p3_w2[1] * p3_1 + p3_w2[2] * self.dnsample(p2_2))
        p4_2 = self.conv4_dn(p4_w2[0] * p4_0 + p4_w2[1] * self.dnsample(p3_2))

        return p1_2, p2_2, p3_2, p4_2


class EEGE(nn.Module):

    def __init__(self, num_classes=2):
        super(EEGE, self).__init__()

        # common modules
        self.gap = nn.AdaptiveAvgPool2d(1)

        # region selection
        self.backbone_region = list(models.resnet18().children())[:7]
        self.attention_layer = bot.BottleStack(dim=256, fmap_size=14, dim_out=512, proj_factor=4, downsample=True,
                                               heads=4, dim_head=64, rel_pos_emb=True, activation=nn.ReLU())
        self.region_selector = nn.Sequential(
            *self.backbone_region,
            self.attention_layer,
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Upsample(224)
        )

        # gaze estimation - feature extraction
        self.backbone_gaze = list(models.resnet50().children())[:7]
        self.stem = nn.Sequential(*self.backbone_gaze[:4])
        self.block_1 = self.backbone_gaze[4]
        self.block_2 = self.backbone_gaze[5]
        self.block_3 = self.backbone_gaze[6]
        self.block_4 = bot.BottleStack(dim=1024, fmap_size=32, dim_out=2048, proj_factor=4, downsample=True,
                                       heads=4, dim_head=128, rel_pos_emb=True, activation=nn.ReLU())

        # gaze estimation - feature pyramid
        self.fpn = nn.Sequential(*[BiFPN(64) for _ in range(2)])
        self.conv1 = nn.Conv2d(256, out_channels=64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, out_channels=64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(1024, out_channels=64, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(2048, out_channels=64, kernel_size=1, stride=1)

        # gaze estimation - prediction
        self.hints = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Upsample(224)
        )
        self.prediction_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # a_maps = self.region_selector(x)
        # x = x * a_maps
        feature_0 = self.stem(x)
        feature_1 = self.block_1(feature_0)
        feature_2 = self.block_2(feature_1)
        feature_3 = self.block_3(feature_2)
        feature_4 = self.block_4(feature_3)
        # a_targets = self.hints(feature_4)
        features = self.conv1(feature_1), self.conv2(feature_2), self.conv3(feature_3), self.conv4(feature_4)
        features = self.fpn(features)
        features = torch.cat([self.gap(f) for f in features], dim=1)
        predictions = self.prediction_head(features)
        return predictions
