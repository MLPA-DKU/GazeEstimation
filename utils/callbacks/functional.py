import os
import os.path
import shutil
import torch


def save_checkpoint(state, path, filename='model.pth', is_best=False, topnotch='best_model.pth'):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, topnotch))


def visualize_progress(rate, length=40, symbol='█', whitespace=' '):
    return f"|{symbol * int(length * rate) + whitespace * (length - int(length * rate))}|"
