import torch

from . import functional as F


class AngleError:

    def __init__(self):
        self.ret = None

    def __call__(self, inputs, targets):
        self.ret = F.angle_difference_rad2deg(inputs, targets)
        return torch.mean(self.ret)


class AngleErrorGaze360:

    def __call__(self, inputs, targets):
        return F.angle_difference_gaze360(inputs, targets)
