

class Callbacks:

    def __init__(self, verbose):
        self.verbose = verbose

    def epoch_begin(self):
        pass

    def epoch_end(self):
        pass

    def train_begin(self):
        pass

    def train_end(self):
        pass

    def validation_begin(self):
        pass

    def validation_end(self):
        pass

    def inference_begin(self):
        pass

    def inference_end(self):
        pass
