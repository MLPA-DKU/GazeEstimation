

class ResCapture:

    def __init__(self, tqdm, args, mode):
        self.tqdm = tqdm
        self.args = args
        self.mode = mode
        self.mode_dict = {'train': 'TRAIN', 'valid': 'VALID', 'infer': 'INFER'}

        self.tqdm.set_description(f'{self.mode_dict[self.mode]} EPOCH[{self.args.epoch + 1:4d}/{self.args.epochs:4d}]')
        self.tqdm.bar_format = '{l_bar}{bar}| BATCH[{n_fmt}/{total_fmt}] ETA: {elapsed}<{remaining}{postfix}'

        self.losses = []
        self.scores = []

    def __call__(self, loss, score):
        self.losses.append(loss)
        self.scores.append(score)

        loss_str = f'{self.mode}_loss'
        score_str = f'{self.mode}_score'

        self.args.writer.add_scalar(loss_str, loss, self.args.epoch * len(self.tqdm) + self.args.idx)
        self.args.writer.add_scalar(score_str, score, self.args.epoch * len(self.tqdm) + self.args.idx)

        self.tqdm.set_postfix_str(f'{loss_str}: {loss:.3f}, {score_str}: {score:.3f}')

    def results(self):
        return self.losses, self.scores
