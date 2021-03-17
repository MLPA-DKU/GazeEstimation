import os
import os.path
import torch


def make_directory_available(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def save_checkpoint(checkpoint, checkpoint_name, save_dir):
    p = os.path.join(save_dir, checkpoint_name)
    torch.save(checkpoint, p)


# {head}_{epoch}_{????:>0{len(str(epoch))}d}_{loss}_{?.???:.3f}
def model_name(head, **kwargs):
    return head + '_' + '_'.join([f'{k}_{v}' for k, v in kwargs.items()]) + '.pth'
