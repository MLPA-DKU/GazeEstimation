import torch

from . import fuctional as F


class AngularError:

    def __init__(
            self,
            reduction: str = 'mean',
        ):
        self.reduction = reduction

    def __call__(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        return F.angular_error(inputs, targets, reduction=self.reduction)
