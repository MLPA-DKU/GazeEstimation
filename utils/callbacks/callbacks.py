# from . import functional as F


class CheckPoint:

    def __init__(self, filepath, monitor, save_best_only=False, save_weights_only=False, mode='auto', verbose=0):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.verbose = verbose


class EarlyStop:

    def __init__(self, monitor, delta=0, patience=0, baseline=None, mode='auto', verbose=0):
        self.monitor = monitor
        self.delta = delta
        self.patience = patience
        self.baseline = baseline
        self.mode = mode
        self.verbose = verbose


class TensorBoard:

    def __init__(self, writer):
        self.writer = writer

    def __call__(self, loss, score):
        pass
