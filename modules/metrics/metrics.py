from . import fuctional as F


class AngularError:

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, inputs, targets):
        return F.angular_error(inputs, targets, reduction=self.reduction)
