import torch
import torch.nn as nn

from . import fuctional as F


class _Metric(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(_Metric, self).__init__()
        self.reduction = reduction


class AngularError(_Metric):

    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(AngularError, self).__init__(reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.angular_error(inputs, targets, reduction=self.reduction)
