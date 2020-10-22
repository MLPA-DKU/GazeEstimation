import collections
import time

from utils.callback import functional as F


class Glance:

    def __init__(self, epochs, batches):
        self.epoch = 1
        self.batch = 0
        self.epochs = epochs
        self.batches = batches

        self.srt = time.perf_counter()
        self.tic = time.perf_counter()
        self.elapsed = None
        self.exec_time = None

        self.batch_history = collections.defaultdict(list)
        self.batch_progress_rate = None

    def step(self):
        self.toc = time.perf_counter()
        self.elapsed = self.toc - self.srt
        self.exec_time = self.toc - self.tic
        self.tic = self.toc

        if self.batch == self.batches:
            self.batch = 0
            self.epoch += 1
        self.batch += 1
        self.batch_progress_rate = self.batch / self.batches

    def collect(self, **kwargs):
        for k, v in kwargs.items():
            self.batch_history[k].append(v)

    def sync(self):
        pass


def convert_to_time_format(periods):
    ss = periods % 60
    mm = periods // 60 % 60
    hh = periods // 60 // 60 % 24
    dd = periods // 60 // 60 // 24
    tt = f'{mm:02d}:{ss:02d}' if hh == 0 else f'{hh:02d}:{mm:02d}:{ss:02d}'
    tt = f'{tt}' if dd ==0 else f'{dd}d {tt}'
    return tt


if __name__ == '__main__':
    import random

    epochs = 100
    batches = 50000

    glance = Glance(epochs, batches)
    for i in range(epochs):
        for j in range(batches):
            glance.step()
            print(f'\r(train) '
                  f'Epoch[{glance.epoch:>{len(str(glance.epochs))}d}/{glance.epochs}] '
                  f'{100 * glance.batch_progress_rate:5.1f}% '
                  f'{F.visualize_progress(glance.batch_progress_rate)} '
                  f'{glance.batch:>{len(str(glance.batches))}d}/{glance.batches} '
                  f'ETA: {convert_to_time_format(int(glance.elapsed))}<{convert_to_time_format(int(glance.exec_time * (glance.batches - glance.batch)))} :: '
                  f'Loss: {random.random():.3f}, '
                  f'Acc@1: {100 * random.random():>5.1f}%', end='')
            time.sleep(2)
        print()
