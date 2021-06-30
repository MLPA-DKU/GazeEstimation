import numpy as np
import torch.nn as nn
import torch.utils.data as loader
import torchvision.transforms as transforms

import datasets
import models
import modules
import utils

# global settings
device = utils.auto_device()
epochs = 1000

# dataset option
root = '/mnt/datasets/Gaze/Gaze360'
save = '/mnt/saves/Gaze'

# dataloader option
batch_size = 32
num_workers = 16


def main():

    print('\rInitializing...', end='')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    trainset = datasets.Gaze360(root=root, train=True, transform=transform, mode='frame')
    validset = datasets.Gaze360(root=root, train=False, transform=transform, mode='frame')
    trainloader = loader.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = loader.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = models.XiT([64, 64, 64, 128, 128, 256, 256], depth=6)
    model.to(device)

    optimizer = modules.Lookahead(modules.RAdam(model.parameters()))
    criterion = nn.MSELoss()
    evaluator = modules.AngularError()


if __name__ == '__main__':
    main()
