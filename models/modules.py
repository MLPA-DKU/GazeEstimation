from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class SpatialAttention(nn.Module):

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1x1(x)
        return x


def view(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


image = Image.open('../utils/visualization/example.jpg').convert('RGB')
view(image)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

image = transform(image)

down_channels = SpatialAttention(3)
image = down_channels(image.unsqueeze(0)).squeeze(0)

view(image.detach().numpy().transpose(1, 2, 0))

breakpoint()
