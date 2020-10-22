import os
import os.path
import shutil
import torch


def save_checkpoint(state, filepath, is_best=False, topnotch='best_model.pth.tar'):
    if not os.path.exists(os.path.dirname(filepath)):
        os.mkdir(os.path.dirname(filepath))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(os.path.dirname(filepath), topnotch))


def visualize_progress(rate, length=40, symbol='█', whitespace=' '):
    return f"|{symbol * int(length * rate) + whitespace * (length - int(length * rate))}|"
