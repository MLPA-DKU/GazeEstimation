import numpy as np

# from . import functional as F


class CheckPoint:

    def __init__(self, filepath, monitor, save_best_only=False, save_weights_only=False, mode='auto', verbose=0):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.verbose = verbose


class EarlyStopping:

    def __init__(self, delta=0, patience=0, verbose=0):
        self.delta = delta
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.monitor = np.Inf
        self.early_stop = False
        self.record_breaking = False
        self.message = None

    def __call__(self, monitor):
        if monitor >= self.monitor - self.delta:
            self.counter += 1
            self.record_breaking = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.monitor = min(self.monitor, monitor)
            self.counter = 0
            self.record_breaking = True

        if self.verbose > 0:
            self.message = f'ESC[{self.counter:{len(str(self.patience))}d}/{self.patience:}]'


class TensorBoard:

    def __init__(self, writer):
        self.writer = writer

    def __call__(self, loss, score):
        pass


if __name__ == '__main__':
    callback = EarlyStopping(patience=3, verbose=1)

    for i in range(10):
        loss = 0.1
        if i > 1:
            loss = 0.01
        callback(loss)
        print(callback.message)
        if callback.early_stop:
            break
