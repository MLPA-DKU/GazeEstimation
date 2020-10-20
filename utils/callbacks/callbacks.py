import os.path
import shutil
import numpy as np
import torch


class ResCapture:

    def __init__(self, tqdm, args, mode):
        self.tqdm = tqdm
        self.args = args
        self.mode = mode
        self.mode_dict = {'train': 'TRAIN', 'valid': 'VALID', 'infer': 'INFER'}

        self.tqdm.set_description(f'{self.mode_dict[self.mode]} EPOCH[{self.args.epoch + 1:4d}/{self.args.epochs:4d}]')
        self.tqdm.bar_format = '{l_bar}{bar:40}| BATCH[{n_fmt}/{total_fmt}] ETA: {elapsed}<{remaining}{postfix}'

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


class Checkpointer:

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.best_score = np.inf

    def __call__(self, model, score):
        torch.save(model, os.path.join(self.path, self.name))
        is_best = self.best_score > score
        self.best_score = min(self.best_score, score)
        if is_best:
            shutil.copyfile(os.path.join(self.path, self.name), os.path.join(self.path, 'model_best.pth'))
