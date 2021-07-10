from typing import BinaryIO, IO, Union

import os
import logging

import torch.utils.tensorboard


def setup_logger(level=logging.INFO):
    head = '\r[%(asctime)-15s] (%(filename)s:line %(lineno)d) %(name)s:%(levelname)s :: %(message)s'
    logging.basicConfig(format=head, level=level)


def enable_overlapping_logging():
    logging.StreamHandler.terminator = ''


def disable_overlapping_logging():
    logging.StreamHandler.terminator = '\n'


def create_tensorboard_writer(f):
    return TensorboardHandler(f)


class TensorboardHandler:

    def __init__(
            self,
            f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        ):
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=f)

    def log(
            self,
            tag: str,
            value,
            global_step: int,
        ) -> None:
        value = value.item() if isinstance(value, torch.Tensor) else value
        self.writer.add_scalar(tag, value, global_step)
