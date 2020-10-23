import time


class TickTack:

    def __init__(self):
        self.tick = time.perf_counter()
        self.tack = None
        self.tock = None

    def __call__(self):
        self.tack = time.perf_counter()
        self.tock = self.tack - self.tick
        self.tick = self.tack
