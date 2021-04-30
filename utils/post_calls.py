import gc
import os
import uuid

import numpy as np
import torch
import torch.utils.tensorboard


class TerminatorModule:

    def __init__(self, monitor=None, patience=0, delta=0):
        self.monitor = monitor
        self.patience = patience
        self.delta = delta

        self.best_score = np.inf
        self.counter = 1

    def __call__(self):
        if self.counter == self.patience:
            self.__terminator__()
        self.counter = 1 if self.__discriminator__() else self.counter + 1

    def __discriminator__(self):
        is_best = True if self.monitor < self.best_score - self.delta else False
        self.best_score = min(self.monitor, self.best_score - self.delta)
        return is_best

    def __terminator__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        quit()


class CheckpointModule:

    base_folder = 'checkpoint'

    def __init__(self, path, unique_id, obj):
        self.path = os.path.join([path, self.base_folder, f'{self.base_folder}.{unique_id}'])
        if os.path.exists(self.path):
            os.makedirs(self.path)
        self.target = obj

    def __call__(self):
        pass


class IntegratedManagementModule:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
