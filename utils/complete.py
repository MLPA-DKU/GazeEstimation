from typing import Union
import os
import os.path
import uuid

import numpy as np
import torch
import torch.utils.tensorboard


class R6FolderManager:

    base_folder = NotImplemented

    def __init__(self, f: Union[str, os.PathLike], uniq: Union[str, int, uuid.UUID]):
        self.f = os.path.join(f, self.base_folder, f'{self.base_folder}.{uniq}' if uniq else f'{self.base_folder}')
        self.f = os.path.abspath(self.f)
        if not os.path.exists(self.f):
            os.makedirs(self.f)
        self.uniq = uniq

class R6Tensorboard(R6FolderManager):

    def __init__(self, f, uniq=None):
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
