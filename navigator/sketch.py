import numpy as np


class AverageMeter:

    def __init__(self, name=None, format_spec='.f'):
        self.name = name
        self.format_spec = format_spec
        self.recorder = []
        self.mean = None

    def __str__(self):
        return format(self.mean, self.format_spec)

    def collect(self, value):
        self.recorder.append(value)
        self.mean = np.nanmean(self.recorder)

    def initialize(self):
        self.recorder = []
        self.mean = None


class ScoreMeter:

    def __init__(self, delta=0):
        self.delta = delta
        self.best_score = np.inf
        self.is_best = False

    def __call__(self, score):
        if self.best_score - self.delta > score:
            self.is_best = True
        else:
            self.is_best = False
        self.best_score = min(self.best_score - self.delta, score)


class CircuitBreaker:

    # early stopper

    def __init__(self, monitor=None, delta=0, patience=0, baseline=None, verbose=0):
        self.monitor = monitor
        self.patience = patience
        self.baseline = baseline
        self.verbose = verbose

        self.score_meter = ScoreMeter(delta=delta)
        self.counter = 1

    def __call__(self, monitor=None):

        if self.counter == self.patience:
            quit()

        value = self.monitor[-1] if self.monitor is not None else monitor
        self.score_meter(value)
        if self.baseline:
            self.score_meter.is_best = False if value < self.baseline else self.score_meter.is_best

        self.counter = 1 if self.score_meter.is_best else self.counter + 1

        if self.verbose > 0:
            print(f'Circuit breaker counter [{self.counter}/{self.patience}]')


if __name__ == '__main__':
    import time

    losses = []
    # cb = CircuitBreaker(losses, patience=5, verbose=1)
    cb = CircuitBreaker(patience=5, verbose=1)

    for _ in range(20):
        # losses.append(0.5)
        cb(0.5)
        time.sleep(0.1)
