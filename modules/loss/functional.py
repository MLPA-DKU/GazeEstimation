import torch


def pin_ball_loss(inputs, targets, variances, quantile_10, quantile_90, reduction):
    area_10 = targets - (inputs - variances)
    area_90 = targets - (inputs + variances)
    loss_10 = torch.max(quantile_10 * area_10, (quantile_10 - 1) * area_10)
    loss_90 = torch.max(quantile_90 * area_10, (quantile_90 - 1) * area_90)
    loss_10 = torch.mean(loss_10) if reduction is 'mean' else torch.sum(loss_10)
    loss_90 = torch.mean(loss_90) if reduction is 'mean' else torch.sum(loss_90)
    return loss_10 + loss_90
