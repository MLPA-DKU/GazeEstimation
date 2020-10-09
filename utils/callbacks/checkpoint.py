

class Checkpoint:

    def __init__(self):
        self.best_score = 0
        self.is_best = True

    def __call__(self, *args, **kwargs):
        pass
