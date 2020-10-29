

class History:

    TRAIN = 'train'
    VALID = 'valid'

    def __init__(self, epochs, batches):
        self.sequence = self.TRAIN
        self.epoch = 1
        self.batch = 0
        self.epochs = epochs
        self.batches = batches

    def train(self):
        self.sequence = self.TRAIN

    def eval(self):
        self.sequence = self.VALID

    def step(self):
        self.epoch += 1

        pass

    def collect(self):
        pass

    def display(self):
        pass


def visualize_progress(rate, length=40, symbol='█', whitespace=' '):
    return f"|{symbol * int(length * rate) + whitespace * (length - int(length * rate))}|"
