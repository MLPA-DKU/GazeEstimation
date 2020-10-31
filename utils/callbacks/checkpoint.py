import os
import os.path
import shutil
import torch


class CheckPoint:

    def __init__(self, filepath, save_best_only=False, save_weights_only=False, verbose=0):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        if not os.path.exists(os.path.dirname(self.filepath)):
            os.mkdir(os.path.dirname(self.filepath))

        if self.save_best_only:
            self.filepath = os.path.join(os.path.dirname(self.filepath), 'model_best.pth.tar')

    def __call__(self, state, is_best):
        if self.verbose > 0:
            print('\npreparing checkpoint to save...')

        if self.save_weights_only:
            for k in ['model', 'optimizer']:
                if k in state:
                    module = state[k]
                    state[k] = module.state_dict() if module is not None else None

        if self.save_best_only:
            if is_best:
                torch.save(state, self.filepath)
        else:
            torch.save(state, self.filepath)
            if is_best:
                shutil.copyfile(self.filepath, os.path.join(os.path.dirname(self.filepath), 'model_best.pth.tar'))

        if self.verbose > 0:
            print(f'\n...saving checkpoint at {self.filepath} successfully')
