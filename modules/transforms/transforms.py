from . import functional as F


class DeNormalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return F.denormalize(tensor, self.mean, self.std)
