import os
import pathlib
from typing import Union
import yaml

import utils


class ConfigurationManager:

    def __init__(self, config: Union[str, pathlib.Path, os.PathLike]):
        self.config = self.config_parser(config)
        self.device = self.config['experiment']['args']['device']
        self.device = utils.auto_device() if self.device == 'auto' else self.device

    @staticmethod
    def config_parser(f):
        with open(f) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    @staticmethod
    def __materialize__(config, name, module, *args, **kwargs):
        config_args = config[name]['args']
        config_args = dict(config_args) if config_args is not None else dict()
        assert all([k not in config_args for k in kwargs]), 'configuration takes precedence. overwriting isn`t allowed.'
        config_args.update(kwargs)
        return getattr(module, config[name]['name'])(*args, **config_args)

    def materialize(self, name, module, *args, **kwargs):
        return self.__materialize__(self.config, name, module, *args, **kwargs)


if __name__ == '__main__':
    import torch.utils.data
    import torchvision.transforms as transforms
    import datasets
    import models
    import modules

    conf = ConfigurationManager('temp.yaml')

    print('\rInitializing...', end='')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    trainset = conf.materialize('dataset', datasets, train=True, transform=transform)
    validset = conf.materialize('dataset', datasets, train=False, transform=transform)
    trainloader = conf.materialize('dataloader', torch.utils.data, trainset, shuffle=True)
    validloader = conf.materialize('dataloader', torch.utils.data, validset, shuffle=False)

    model = conf.materialize('model', models)
    model.to(conf.device)

    optimizer = modules.Lookahead(conf.materialize('optimizer', modules, model.parameters()))
    criterion = conf.materialize('criterion', torch.nn)
    evaluator = conf.materialize('evaluator', modules)

    breakpoint()
