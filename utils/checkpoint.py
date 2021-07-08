from typing import BinaryIO, Dict, IO, Union

import logging
import os
import torch


def create_checkpoint_handler(checkpoint_obj, f):
    return CheckpointHandler(checkpoint_obj, f)


class CheckpointHandler:

    def __init__(
            self,
            obj: Dict[str, int, torch.nn.Module, torch.optim.Optimizer],
            f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        ):
        self.obj = obj
        self.f = f

    def save(self) -> None:
        try:
            logging.debug(f'trying to save checkpoint at {self.f}...')
            torch.save(self.obj, self.f)
            logging.debug(f'saving checkpoint at {self.f} successfully')
        except Exception as e:
            logging.error(f'error occurs when saving checkpoint at {self.f} by "{e}"')

    def load(self):
        try:
            logging.debug(f'trying to load checkpoint from {self.f}...')
            ...
            logging.debug(f'loading checkpoint from {self.f} successfully')
        except Exception as e:
            logging.error(f'error occurs when loading checkpoint from {self.f} by "{e}"')
