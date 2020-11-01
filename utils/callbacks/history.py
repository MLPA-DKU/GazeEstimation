

class History:

    def __init__(self, batches):
        self.batch = 0
        self.epoch = 0
        self.batches = batches

    def step(self):
        pass

    def collect(self):
        pass

    def display(self):
        pass

    def __render__(self, rate, length=40, symbol='█', whitespace=' '):
        return f"|{symbol * int(length * rate) + whitespace * (length - int(length * rate))}|"
