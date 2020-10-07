import math
import torch


class AngleAccuracy:

    def __call__(self, inputs, targets):

        with torch.no_grad():
            pred_x = -1 * torch.cos(inputs[:, 0]) * torch.sin(inputs[:, 1])
            pred_y = -1 * torch.sin(inputs[:, 0])
            pred_z = -1 * torch.cos(inputs[:, 0]) * torch.cos(inputs[:, 1])
            pred_n = torch.sqrt(pred_x ** 2 + pred_y ** 2 + pred_z ** 2)

            true_x = -1 * torch.cos(targets[:, 0]) * torch.sin(targets[:, 1])
            true_y = -1 * torch.sin(targets[:, 0])
            true_z = -1 * torch.cos(targets[:, 0]) * torch.cos(targets[:, 1])
            true_n = torch.sqrt(true_x ** 2 + true_y ** 2 + true_z ** 2)

            angle_rad = (pred_x * true_x + pred_y * true_y + pred_z * true_z) / (true_n * pred_n)
            return torch.mean(torch.acos(angle_rad) * 180.0 / math.pi)


class ClassificationAccuracy:

    def __call__(self, inputs, targets, topk=(1,)):

        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)

            _, pred = inputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
