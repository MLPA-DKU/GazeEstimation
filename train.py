import torch.nn as nn
import torch.utils.data as loader
import torchvision.transforms as transforms

import datasets
import models


def main(config):

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

    model = ...

    optimizer = ...
    criterion = nn.MSELoss()
    evaluator = ...


if __name__ == '__main__':
    main()
