import collections


class ScoreContainer:

    def __init__(self, name=None, format_spec='.f'):
        self.name = name
        self.format_spec = format_spec
        self.values = []
        self.mean = None

    def __str__(self):
        return format(self.mean, self.format_spec)

    def step(self, value):
        self.values.append(value)
        self.mean = np.nanmean(self.values)

    def initialize(self):
        self.values = []
        self.mean = None


class VirtualModule:

    def __init__(self):
        self.handlers = collections.defaultdict(ScoreContainer)

    def epoch_start(self):
        self.handlers = collections.defaultdict(ScoreContainer)

    def epoch_end(self):
        del self.handlers

    def batch_start(self):
        pass

    def batch_end(self):
        pass

    def update(self, name, value):
        self.handlers[name].step(value)


if __name__ == '__main__':

    board = VirtualModule()

    for epoch in range(2):

        board.epoch_start()

        for idx, batch in enumerate(range(2)):

            board.batch_start()

            loss = 0.5
            score = 0.5

            board.update('loss', loss)
            board.update('score', score)

            board.batch_end()

        board.epoch_end()
