
class EarlyStopping:

    def __init__(self, delta=0, patience=0, verbose=0):
        self.delta = delta
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.monitor = np.Inf
        self.early_stop = False
        self.record_breaking = False
        self.message = None

    def __call__(self, monitor):
        if monitor >= self.monitor - self.delta:
            self.counter += 1
            self.record_breaking = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.monitor = min(self.monitor, monitor)
            self.counter = 0
            self.record_breaking = True

        if self.verbose > 0:
            self.message = f'...early stopping count: {self.counter:>{len(str(self.patience))}d}' \
                           f' out of {self.patience:}'
