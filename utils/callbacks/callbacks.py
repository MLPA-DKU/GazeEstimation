from . import functional as F


# (train) Epoch[0000/0000] 000.0% |________________________________________| 0000/0000, ETA: 00:00, Loss: 0.000, Metrics: 0.000
class Glance:

    def __init__(self, sequence, epochs, batches, verbose=1, fmt=':.3f'):
        self.sequence = sequence
        self.epoch = 0
        self.batch = 0
        self.epochs = epochs
        self.batches = batches
        self.verbose = verbose
        self.fmt = fmt

        self.batch_cache = None
        self.batch_progress = None
        self.batch_percents = None
        self.batch_progress_bar = None
        self.batch_terminated = False

    def step(self):
        self.batch += 1
        self.batch_progress = self.batch / self.batches
        self.batch_progress_bar = F.visualize_progress(self.batch_progress)
        if self.batch == self.batches:
            self.batch_terminated = True

    def step_epoch(self):
        self.batch = 0
        self.epoch += 1
