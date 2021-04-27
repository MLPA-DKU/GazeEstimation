import numpy as np


class ScoreCheck:

    def __init__(self, delta=0):
        self.delta = delta
        self.best_score = np.inf

    def __call__(self, score):
        is_best = True if score < self.best_score - self.delta else False
        self.best_score = min(score, self.best_score - self.delta)
        return is_best


class EarlyBird:

    def __init__(self, monitor=None, patience=0, delta=0, verbose=0):
        self.monitor = monitor
        self.patience = patience
        self.is_best = ScoreCheck(delta=delta)
        self.counter = 1
        self.verbose = verbose

    def __call__(self, monitor=None):

        if self.counter == self.patience:
            quit()

        assert (self.monitor is not None or monitor is not None)
        monitor = self.monitor[-1] if monitor is None else monitor
        if self.is_best(monitor):
            self.counter = 1
        else:
            self.counter += 1

        if self.verbose > 0:
            print(f'EarlyBird Counter [{self.counter}/{self.patience}]')


class SavingModule:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass