import torch
import torch.nn as nn
import torchvision.models as models


class SpatialMask(nn.Module):

    def __init__(self):
        super(SpatialMask, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':

    model = models.resnet50()
    modules = list(model.children())[:-2]
    net = nn.Sequential(*modules)

    breakpoint()
