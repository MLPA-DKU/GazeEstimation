import torch
import torch.nn as nn
import torchvision.models as modellib

from models.common.pyramids import BiFPN


class EEGE(nn.Module):

    def __init__(self, num_classes=2):
        super(EEGE, self).__init__()

        # common
        self.gap = nn.AdaptiveAvgPool2d(1)

        # feature extraction
        self.backbone_gaze = list(modellib.resnet50().children())[:8]
        self.stem = nn.Sequential(*self.backbone_gaze[:3])
        self.pool = self.backbone_gaze[3]
        self.block_1 = self.backbone_gaze[4]
        self.block_2 = self.backbone_gaze[5]
        self.block_3 = self.backbone_gaze[6]
        self.block_4 = self.backbone_gaze[7]

        # gaze estimation - feature pyramid
        self.fpn = nn.Sequential(*[BiFPN(64) for _ in range(2)])
        self.conv1 = nn.Conv2d(256, out_channels=64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, out_channels=64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(1024, out_channels=64, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(2048, out_channels=64, kernel_size=1, stride=1)

        # gaze estimation - prediction
        self.prediction_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        feature_0 = self.stem(x)
        feature_1 = self.block_1(self.pool(feature_0))
        feature_2 = self.block_2(feature_1)
        feature_3 = self.block_3(feature_2)
        feature_4 = self.block_4(feature_3)
        features = feature_0, self.conv1(feature_1), self.conv2(feature_2), self.conv3(feature_3), self.conv4(feature_4)
        features = self.fpn(features)
        features = torch.cat([self.gap(f) for f in features], dim=1)
        predictions = self.prediction_head(features)
        return predictions


if __name__ == '__main__':

    model = EEGE()
    inputs = torch.rand((1, 3, 224, 224))
    outputs = model(inputs)

    breakpoint()
