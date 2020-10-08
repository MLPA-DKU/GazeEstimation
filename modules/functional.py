import math
import torch


def angle_difference_rad2deg(inputs, targets):
    with torch.no_grad():
        pred_x = -1 * torch.cos(inputs[:, 0]) * torch.sin(inputs[:, 1])
        pred_y = -1 * torch.sin(inputs[:, 0])
        pred_z = -1 * torch.cos(inputs[:, 0]) * torch.cos(inputs[:, 1])
        pred_n = torch.sqrt(pred_x ** 2 + pred_y ** 2 + pred_z ** 2)

        true_x = -1 * torch.cos(targets[:, 0]) * torch.sin(targets[:, 1])
        true_y = -1 * torch.sin(targets[:, 0])
        true_z = -1 * torch.cos(targets[:, 0]) * torch.cos(targets[:, 1])
        true_n = torch.sqrt(true_x ** 2 + true_y ** 2 + true_z ** 2)

        angle = (pred_x * true_x + pred_y * true_y + pred_z * true_z) / (true_n * pred_n)
        angle = torch.acos(angle) * 180.0 / math.pi
    return angle
