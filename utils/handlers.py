import collections


class TensorboardHandler:

    def __init__(self):
        pass


class MetricTracker:

    def __init__(self):
        self.metric_trackers = collections.defaultdict(list)

    def track(self, tag, value):
        value = value.item() if isinstance(value, torch.Tensor) else value
        self.metric_trackers[tag].append(value)


class StreamHandler:

    def __init__(self):
        pass

    def step(self, tags):
        message = []
        for tag in tags:
            l = self.tracker.metric_trackers[tag]
            tag = ' '.join(tag.split()[1:]) if self.no_header else tag
            message.append(f'{tag}: {l[-1]:{self.format_spec}} ({np.nanmean(l):{self.format_spec}})')
        print(f"\r{' - '.join(message)}", end='' if not self.batch + 1 == batches else '\n')

    def summary(self):
        pass


class SessionHandler:

    def __init__(self, epochs, trainloader, validloader):
        self.epoch = 0
        self.epochs = epochs


if __name__ == '__main__':
    import time
    import numpy as np
    import torch

    epochs = 10
    batches = 100

    streamer = StreamHandler()
    session_tracker = MetricTracker()

    for epoch in range(epochs):

        epoch_tracker = MetricTracker()

        # train
        for batch in range(batches):

            # simulating batch outputs
            loss = torch.rand(1)
            score = torch.rand(1)

            epoch_tracker.track('training loss', loss)
            epoch_tracker.track('training angular error', score)

            # simulating computation
            time.sleep(0.3)

        # valid
        for batch in range(batches):

            # simulating batch outputs
            loss = torch.rand(1)
            score = torch.rand(1)

            epoch_tracker.track('validation loss', loss)
            epoch_tracker.track('validation angular error', score)

            # simulating computation
            time.sleep(0.3)

        for k, v in epoch_tracker.metric_trackers.items():
            session_tracker.track(f'{k} / epoch', np.nanmean(v))
