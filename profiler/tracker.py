import numpy as np


class AverageMeter:

    def __init__(self, name, format_spec='.f'):
        self.name = name
        self.format_spec = format_spec

        self.value = None
        self.values = []
        self.mean = None

    def __str__(self):
        return f'{self.name}: {self.value} (avg: {self.mean})'

    def __call__(self, value):
        self.value = value
        self.values.append(self.value)
        self.value = format(self.value, self.format_spec)
        self.mean = format(np.nanmean(self.values), self.format_spec)
        print(self)


if __name__ == '__main__':
    import time
    import random

    timer = AverageMeter('time', '.3f')
    for _ in range(20):
        timer(random.random())
        time.sleep(0.1)
