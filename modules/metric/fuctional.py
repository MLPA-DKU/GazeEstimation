import numpy as np
import torch
import torch.nn.functional as F


def angular_error(inputs, targets, reduction='mean'):

    def convert_to_cartesian(vector):
        x = -1 * torch.cos(vector[:, 0]) * torch.sin(vector[:, 1])
        y = -1 * torch.sin(vector[:, 0])
        z = -1 * torch.cos(vector[:, 0]) * torch.cos(vector[:, 1])
        return torch.stack((x, y, z), dim=1)

    with torch.no_grad():
        inputs = convert_to_cartesian(inputs)
        targets = convert_to_cartesian(targets)
        product = F.cosine_similarity(inputs, targets, dim=1, eps=1e-8)
        product = torch.acos(product)
        product = 180 * product / np.pi
        product = torch.mean(product) if reduction == 'mean' else torch.sum(product)

    return product
