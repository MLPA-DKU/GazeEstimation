import time


class Glance:

    def __init__(self, sequence, epochs, batches):
        # progress
        self.sequence = sequence
        self.epoch = 1
        self.batch = 0
        self.epochs = epochs
        self.batches = batches
        self.batch_progress_rate = None

        # timer
        self.srt = time.perf_counter()
        self.tic = time.perf_counter()
        self.toc = None
        self.elapsed = None
        self.remains = None
        self.exec_time = None

        # printer
        self.write_buffer = None

    def step(self):
        if self.batch == self.batches:
            self.srt = time.perf_counter()
            self.batch = 0
            self.epoch += 1
        self.tictoc()
        self.batch += 1
        self.batch_progress_rate = self.batch / self.batches
        self.update_buffer()

    def tictoc(self):
        self.toc = time.perf_counter()
        self.elapsed = self.toc - self.srt
        self.exec_time = self.toc - self.tic
        self.remains = self.exec_time * (self.batches - self.batch)
        self.tic = self.toc

    def update_buffer(self):
        self.write_buffer = ' '.join([
            f'\r({self.sequence})',
            f'Epoch[{self.epoch:>{len(str(self.epochs))}d}/{self.epochs}]',
            f'{100 * self.batch_progress_rate:5.1f}%',
            f'{visualize_progress(self.batch_progress_rate)}',
            f'{self.batch:>{len(str(self.batches))}d}/{self.batches}',
            f'ETA: {timestamp(int(self.elapsed))}<{timestamp(int(self.remains))} ::'
        ])


def timestamp(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def visualize_progress(rate, length=40, symbol='█', whitespace=' '):
    return f"|{symbol * int(length * rate) + whitespace * (length - int(length * rate))}|"


if __name__ == '__main__':
    import random

    epochs = 5
    batches = 10

    tg = Glance('train', epochs, batches)
    vg = Glance('valid', epochs, batches)
    for i in range(epochs):
        for j in range(batches):
            tg.step()
            print(f'{tg.write_buffer} '
                  f'Loss: {random.random():.3f}, '
                  f'Acc@1: {100 * random.random():>5.1f}%', end='')
            time.sleep(0.1)
        print()
        for j in range(batches):
            vg.step()
            print(f'{vg.write_buffer} '
                  f'Loss: {random.random():.3f}, '
                  f'Acc@1: {100 * random.random():>5.1f}%', end='')
            time.sleep(0.1)
        print()
