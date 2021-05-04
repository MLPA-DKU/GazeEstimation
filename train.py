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

# dataloader option
batch_size = 64
num_workers = 16


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    trainset = datasets.Gaze360(root=root, train=True, transform=transform, mode='image')
    validset = datasets.Gaze360(root=root, train=False, transform=transform, mode='image')
    trainloader = loader.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = loader.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = models.EEGE()
    model.to(device)

    optimizer = modules.Lookahead(modules.RAdam(model.parameters()))
    criterion = nn.MSELoss()
    evaluator = modules.AngularError()

    r6 = ...

    for epoch in range(epochs):
        train(trainloader, model, optimizer, criterion, evaluator, r6)
        valid(validloader, model, criterion, evaluator, r6)
        r6.end_epoch()


def train(dataloader, model, optimizer, criterion, evaluator, r6):

    model.train()
    r6.train()
    for idx, batch in enumerate(dataloader):
        _, targets, outputs, loss = modules.update(batch, model, optimizer, criterion, device=device)
        score = evaluator(outputs, targets)
        r6.add_metric('loss', loss)
        r6.add_metric('angular error', score)
        r6.end_batch_summary()


def valid(dataloader, model, criterion, evaluator, r6):

    model.eval()
    r6.eval()
    for idx, batch in enumerate(dataloader):
        loss, score = modules.evaluate(batch, model, criterion, evaluator, device=device)
        r6.add_metric('loss', loss)
        r6.add_metric('angular error', score)
        r6.end_batch_summary()
    r6.end_epoch_summary()


if __name__ == '__main__':
    main()
