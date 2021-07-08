import logging

import torch.nn as nn
import torch.utils.data as loader
import torchvision.transforms as transforms

import datasets
import models
import modules
import utils


# temp variable
root = '/mnt/datasets/Gaze/Gaze360'
num_workers = 8
epochs = 100


def bootstrap_dataloader(updater):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    trainset = datasets.Gaze360(root=root, train=True, transform=transform, mode='frame')
    validset = datasets.Gaze360(root=root, train=False, transform=transform, mode='frame')

    batch_size = utils.auto_batch_size(trainset, updater)

    trainloader = loader.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = loader.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader

def main():

    modules.setup_logger(level=logging.DEBUG)
    device = utils.auto_device()

    model = models.GazeCT([64, 64, 64, 128, 128, 256, 256], depth=6)
    model.to(device)

    optimizer = modules.Lookahead(modules.RAdam(model.parameters()))
    criterion = nn.MSELoss()
    evaluator = modules.AngularError()

    trainer = modules.Engine(modules.update(model, optimizer, criterion, device), [criterion, evaluator], train=True)
    validator = modules.Engine(modules.evaluate(model, device), [criterion, evaluator], train=False)
    trainloader, validloader = bootstrap_dataloader(trainer.process_fn)

    for epoch in range(epochs):
        trainer(trainloader)
        validator(validloader)


if __name__ == '__main__':
    main()
