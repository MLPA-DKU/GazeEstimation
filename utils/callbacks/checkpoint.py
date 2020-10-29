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


class CheckPoint:

    def __init__(self, filepath, save_best_only=False, save_weights_only=False, verbose=0):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

    def __call__(self, state, is_best):
        if self.verbose > 0:
            print('\npreparing checkpoint to save...')

        if self.save_weights_only:
            for k in ['model', 'optimizer']:
                if k in state:
                    module = state[k]
                    state[k] = module.state_dict() if module is not None else None

        if self.save_best_only:
            self.filepath = os.path.join(os.path.dirname(self.filepath), 'model.pth.tar')
            if is_best:
                save_checkpoint(state, self.filepath, is_best=False)
                if self.verbose > 0:
                    print(f'\n...saving checkpoint at {self.filepath} successfully')
        else:
            save_checkpoint(state, self.filepath, is_best=is_best)
            if self.verbose > 0:
                print(f'\n...saving checkpoint at {self.filepath} successfully')
