import time


class Clockwork:

    def __init__(self):
        self.initial_time = time.perf_counter()
        self.tick = time.perf_counter()
        self.tack = None
        self.elapsed = None
        self.iteration_time = None

    def __call__(self):
        self.tack = time.perf_counter()
        self.elapsed = self.tack - self.initial_time
        self.initial_time = self.tack - self.tick
        self.tick = self.tack

    def reset(self):
        self.tick = time.perf_counter()
        self.tack = None
        self.elapsed = None
        self.iteration_time = None
