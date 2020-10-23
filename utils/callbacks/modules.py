import os
import os.path
import numpy as np

from . import functional as F


class CheckPoint:

    def __init__(self, filepath, save_best_only=False, save_weights_only=False, verbose=0):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.message = None

    def __call__(self, obj, is_best):
        if self.save_weights_only:
            for k in ['model', 'optimizer']:
                if k in obj:
                    module = obj[k]
                    obj[k] = module.state_dict() if module is not None else None

        if self.save_best_only:
            self.filepath = os.path.join(os.path.dirname(self.filepath), 'model.pth.tar')
            if is_best:
                F.save_checkpoint(obj, self.filepath, is_best=False)
                if self.verbose > 0:
                    self.message = f'...saving checkpoint successfully'
        else:
            F.save_checkpoint(obj, self.filepath, is_best=is_best)
            if self.verbose > 0:
                self.message = f'...saving checkpoint successfully'


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
            self.message = f'...early stopping count: {self.counter:>{len(str(self.patience))}d}' \
                           f' out of {self.patience:}'
