from typing import Union
import os
import os.path
import uuid
import torch
import torch.nn as nn
import torch.optim as optim


class R6CheckpointBlueprint:

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.kwargs = kwargs

        self.obj = {'model': self.model, 'optimizer': self.optimizer}
        for k, v in self.kwargs.items():
            self.obj[k] = v

    def __call__(self, **kwargs):
        obj = self.obj
        for k, v in kwargs.items():
            obj[k] = v
        return {k: v for k, v in sorted(obj.items())}


class R6FolderManager:

    base_folder = NotImplemented

    def __init__(self, f: Union[str, os.PathLike], uniq: Union[str, int, uuid.UUID]):
        self.f = os.path.join(f, self.base_folder, f'{self.base_folder}.{uniq}' if uniq else f'{self.base_folder}')
        self.f = os.path.abspath(self.f)
        if not os.path.exists(self.f):
            os.makedirs(self.f)
        self.uniq = uniq


# TODO: PyTorch Integrated Checkpoint Module - Inspired by CheckFreq from Microsoft Project Fiddle
class R6Checkpoint(R6FolderManager):

    def __init__(self, obj, f, uniq=None):
        self.base_folder = 'checkpoint'
        super(R6Checkpoint, self).__init__(f=f, uniq=uniq)

        self.obj = obj

    def __tape__(self, suffix):
        return f'checkpoint.R6.{self.uniq}.{suffix}.pth'

    def __save__(self, suffix):
        torch.save(self.obj, os.path.join(self.f, self.__tape__(suffix)))

    def __load__(self, suffix):
        return torch.load(os.path.join(self.f, self.__tape__(suffix)))

    def save(self, suffix):
        self.__save__(suffix)

    def load(self, suffix):
        return self.__load__(suffix)


# if __name__ == '__main__':
#     import models
#     import modules
#
#     model = models.EEGE()
#     optimizer = modules.Lookahead(modules.RAdam(model.parameters()))
#
#     checkpoint_blueprint = R6CheckpointBlueprint(model, optimizer)
#     checkpoint_module = R6Checkpoint(checkpoint_blueprint, ...)
#
#     breakpoint()
