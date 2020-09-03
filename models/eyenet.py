import torch
import torch.nn as nn


class EyeNet(nn.Module):

    def __init__(self, features, mid_out=False):
        super(EyeNet, self).__init__()
        self.features = features
        self.combiner = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.regression = nn.Sequential(
            nn.Linear(514, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mid_out = mid_out

    def forward(self, left, right, headpose):
        left = torch.flatten(self.pool(self.features(left)), 1)
        right = torch.flatten(self.pool(self.features(right)), 1)

        combined = self.combiner(torch.cat((left, right), dim=1))
        combined = torch.cat((combined, headpose), dim=1)

        predict = self.regression(combined)
        return predict, left, right if self.mid_out else predict
