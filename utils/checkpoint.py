from typing import BinaryIO, Dict, IO, Union

import os
import logging

import numpy as np
import torch


def create_checkpoint_handler(
        obj: Dict[str, Union[int, torch.nn.Module, torch.optim.Optimizer]],
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    ):
    return CheckpointHandler(obj, f)


def create_performance_meter():
    return PerformanceMeter()


class CheckpointHandler:

    def __init__(
            self,
            obj: Dict[str, Union[int, torch.nn.Module, torch.optim.Optimizer]],
            f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        ):
        self.obj = obj
        self.f = f

        if not os.path.exists(self.f):
            os.makedirs(self.f)

    def save(self, name: str = 'checkpoint.pth') -> None:
        p = os.path.join(self.f, name)
        try:
            logging.debug(f'trying to save checkpoint at {self.f}...')
            torch.save(self.obj, p)
            logging.debug(f'saving checkpoint at {self.f} successfully')
        except Exception as e:
            logging.error(f'error occurs when saving checkpoint at {self.f} by "{e}"')

    def load(self) -> Dict[str, Union[int, torch.nn.Module, torch.optim.Optimizer]]:
        try:
            logging.debug(f'trying to load checkpoint from {self.f}...')
            raise NotImplementedError
            # logging.debug(f'loading checkpoint from {self.f} successfully')
            # return checkpoint
        except Exception as e:
            logging.error(f'error occurs when loading checkpoint from {self.f} by "{e}"')


class PerformanceMeter:

    def __init__(self):
        self.best_perf = np.inf

    def __call__(self, perf: Union[torch.Tensor, float]) -> bool:
        perf = perf.item() if isinstance(perf, torch.Tensor) else perf
        is_best = True if perf <= self.best_perf else False
        self.best_perf = perf if is_best else self.best_perf
        return is_best
