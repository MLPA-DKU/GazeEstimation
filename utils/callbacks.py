from typing import Union
import os
import os.path
import uuid

import numpy as np
import torch
import torch.utils.tensorboard


class R6SessionManager:

    uniq = str(uuid.uuid4()).split('-')[0]

    def __init__(self, model, optimizer, f, patience=30):
        self.score_board = []
        self.checkpoint_module = R6Checkpoint(model, optimizer, f, self.uniq)
        self.tensorboard_module = R6Tensorboard(f, self.uniq)
        self.early_stopping_module = R6EarlyStopping(self.score_board[-1], patience)


class R6FolderManager:

    base_folder = NotImplemented

    def __init__(self, f: Union[str, os.PathLike], uniq: Union[str, int, uuid.UUID]):
        self.f = os.path.join(f, self.base_folder, f'{self.base_folder}.{uniq}' if uniq else f'{self.base_folder}')
        self.f = os.path.abspath(self.f)
        if not os.path.exists(self.f):
            os.makedirs(self.f)
        self.uniq = uniq


class R6Checkpoint(R6FolderManager):

    # TODO: PyTorch Integrated Checkpoint Module - Inspired by CheckFreq from Microsoft Project Fiddle

    def __init__(self, model, optimizer, f, uniq, **kwargs):
        self.base_folder = 'checkpoint'
        super(R6Checkpoint, self).__init__(f=f, uniq=uniq)

        self.obj = {'model': model, 'optimizer': optimizer}
        self.obj.update(**kwargs)
        self.obj = {k: v for k, v in sorted(self.obj.items())}

    def __tape__(self, suffix):
        return f'checkpoint.R6.{self.uniq}.{suffix}.pth'

    def __save__(self, suffix, **kwargs):
        self.obj.update(**kwargs)
        self.obj = {k: v for k, v in sorted(self.obj.items())}
        torch.save(self.obj, os.path.join(self.f, self.__tape__(suffix)))

    def __load__(self, suffix):
        return torch.load(os.path.join(self.f, self.__tape__(suffix)))

    def save(self, suffix, **kwargs):
        self.__save__(suffix, **kwargs)

    def load(self, suffix):
        return self.__load__(suffix)


class R6Tensorboard(R6FolderManager):

    def __init__(self, f, uniq):
        self.base_folder = 'tensorboard'
        super(R6Tensorboard, self).__init__(f=f, uniq=uniq)

        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.f)


class R6EarlyStopping:

    def __init__(self, monitor=None, patience=0, delta=None):
        self.monitor = monitor
        self.patience = patience
        self.delta = delta if delta is not None else 0

        self.best_score = np.inf
        self.counter = 1

    def __call__(self):
        if self.counter == self.patience:
            quit()
        self.counter = 1 if self.monitor < self.best_score - self.delta else self.counter + 1
        self.best_score = min(self.monitor, self.best_score - self.delta)
