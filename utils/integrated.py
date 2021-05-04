import collections

import numpy as np


class R6:

    def __init__(self):
        self.epoch_manager = None

    def start_epoch(self):
        self.epoch_manager = R6EpochManager()

    def end_epoch(self):
        del self.epoch_manager


class R6GlobalManager:

    def __init__(self):
        self.validation_scores = []


class R6EpochManager:

    """
    construct at the start of epoch, destruct at the end of epoch
    """

    def __init__(self):
        self.metrics = {}

    def add_metric(self, metric, value):
        try:
            self.metrics[metric].metric_values.append(value)
        except KeyError:
            self.metrics[metric] = R6MetricHandler(metric)
            self.metrics[metric].metric_values.append(value)

    def batch_summary(self):
        s = []
        s.extend([str(v) for k, v in self.metrics.items()])
        s = ' - '.join(s)
        print(f'\r{s}', end='')

    def epoch_summary(self):
        pass


class R6MetricHandler:

    def __init__(self, metric, format_spec='.3f'):
        self.metric = metric
        self.metric_values = []
        self.format_spec = format_spec

    def __call__(self, value):
        self.metric_values.append(value)

    def __str__(self):
        return f'{self.metric}: {format(np.nanmean(self.metric_values), self.format_spec)}'


class R6Printer:

    pass


if __name__ == '__main__':
    import time
    import random
    em = R6EpochManager()

    for idx in range(10):
        em.add_metric('loss', random.random())
        em.add_metric('angular error', random.random())
        em.batch_summary()
        time.sleep(0.3)
    breakpoint()
