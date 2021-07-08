from typing import List, Union
import torch


def denorm(
        tensor: torch.Tensor,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
    ) -> torch.Tensor:
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
