import logging

import torch
import torch.utils.data as loader
import torchvision.transforms as transforms
import datasets
import models
import modules
import utils


def setup_logger(level=logging.INFO):
    head = '\r[%(asctime)-15s] (%(filename)s:line %(lineno)d) %(name)s:%(levelname)s :: %(message)s'
    logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(level)


def initializer(config, name, module, *args, **kwargs):
    module_name = config[name]['name']
    module_args = dict(config[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'config file takes precedence. overwriting is not allowed.'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def bootstrapping(config):
    logging.info('bootstrapping sequence started...')
    try:
        model = initializer(config, 'model', models)
        optimizer = initializer(config, 'optimizer', modules)
        criterion = initializer(config, 'criterion', torch.nn)
        evaluator = initializer(config, 'evaluator', modules)
        logging.info('bootstrapping sequence completed successfully')
        return model, optimizer, criterion, evaluator
    except Exception as e:
        logging.error(f'bootstrapping sequence stopped by "{e}"')


def bootstrapping_dataloader(config, updater):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    trainset = initializer(config, 'dataset', datasets, train=True, transform=transform)
    validset = initializer(config, 'dataset', datasets, train=False, transform=transform)
    batch_size = utils.auto_batch_size(trainset, updater)
    trainloader = initializer(config, 'dataloader', loader, trainset, shuffle=True)
    validloader = initializer(config, 'dataloader', loader, validset, shuffle=False)
    return trainloader, validloader
