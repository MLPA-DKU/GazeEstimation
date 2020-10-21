from . import functional as F


class Checkpointer:

    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def __call__(self, state, score):
        F.save_checkpoint(state, self.path, self.filename, True)
