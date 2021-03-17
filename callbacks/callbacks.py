import numpy as np

from . import functional as F


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


def lower_is_better_performance():
    pass


def upper_is_better_performance():
    pass


def is_performance_improved():
    pass


class CheckPoint:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        F.make_directory_available(self.save_dir)

    def __call__(self, checkpoint, is_best, checkpoint_name):
        F.save_checkpoint(checkpoint, checkpoint_name, self.save_dir)
        F.save_checkpoint(checkpoint, 'checkpoint_best.pth', self.save_dir) if is_best else None


if __name__ == '__main__':
    epochs = 1000
    print(F.model_name('checkpoint', epoch=f'{455:>0{len(str(epochs))}d}'))
