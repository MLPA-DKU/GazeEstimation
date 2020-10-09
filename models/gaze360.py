import math
import torch
import torch.nn as nn
import torchvision.models as models


class GazeLSTM(nn.Module):

    def __init__(self, backbone, embedding_dim=256, pretrained=False):
        super(GazeLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.backbone = getattr(models, backbone)(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, bidirectional=True, num_layers=2, batch_first=True)
        self.mapping = nn.Linear(2 * self.embedding_dim, 3)

    def forward(self, x):
        out = self.backbone(x.view((-1, 3) + x.size()[-2:]))
        out = out.view(x.size(0), 7, self.embedding_dim)
        out, _ = self.lstm(out)
        out = out[:, 3, :]
        out = self.mapping(out).view(-1, 3)

        angle = out[:, :2]
        angle[:, 0:1] = math.pi * torch.tanh(angle[:, 0:1])
        angle[:, 1:2] = math.pi / 2 * torch.tanh(angle[:, 1:2])
        var = math.pi * torch.sigmoid(out[:, 0:1])
        var = var.view(-1, 1).expand(var.size(0), 2)
        return angle, var
