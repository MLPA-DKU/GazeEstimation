import torch.nn as nn

from . import functional as F


class PinBallLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(PinBallLoss, self).__init__()
        self.quantile_10 = 0.1
        self.quantile_90 = 1 - self.quantile_10
        self.reduction = reduction

    def forward(self, inputs, targets, variances):
        return F.pin_ball_loss(inputs, targets, variances, self.quantile_10, self.quantile_90, self.reduction)
