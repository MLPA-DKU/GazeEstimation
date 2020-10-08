import os
import json
import shutil
import torch
import matplotlib.pyplot as plt


def read_json(filename):
    with open(os.path.abspath(filename)) as f:
        fig = json.load(f)
    return fig


def print_json(config, indent=''):
    for k, v in config.items():
        if type(config[k]) is DotDict:
            print(indent+f'{k}:')
            print_json(v, indent+'    ')
        else:
            print(indent+f'{k}: {v}')


class DotDict(dict):

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if type(v) is dict:
                self[k] = DotDict(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(e)


class ConfigParser(DotDict):

    def __init__(self, filename):
        super().__init__(read_json(filename))
        print_json(self)

    def initialize_object(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def view(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
