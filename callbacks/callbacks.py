import os
import os.path
import shutil
import numpy as np
import torch


class EarlyStopping:

    def __init__(self, delta=0, patience=0):
        self.delta = delta
        self.patience = patience

        self.counter = 0
        self.monitor = np.inf  # for lower-is-better-performance
        self.early_stop = False

    def __call__(self, monitor):
        if monitor >= self.monitor - self.delta:
            if self.counter == self.patience:
                self.early_stop = True
            self.counter += 1
        else:
            self.monitor = min(self.monitor, monitor)
            self.counter = 0


class CheckPoint:

    def __init__(self, directory):
        self.directory = directory

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def __call__(self, state, is_best, filename):
        filepath = os.path.join(self.directory, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(self.directory, 'model_best.pth'))
