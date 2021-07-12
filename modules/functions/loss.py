import torch.nn as nn


class _Loss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction
