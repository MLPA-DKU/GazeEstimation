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
batch_size = 64
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

    model = models.EEGE()
    model.to(device)

    optimizer = modules.Lookahead(modules.RAdam(model.parameters()))
    criterion = nn.MSELoss()
    evaluator = modules.AngularError()

    r6 = utils.R6SessionManager(model, f=save)

    print('\rInitialization Complete.')

    for epoch in range(epochs):
        train(trainloader, model, optimizer, criterion, evaluator, r6)
        valid(validloader, model, criterion, evaluator, r6)


def train(dataloader, model, optimizer, criterion, evaluator, r6):

    print(f'\rEpoch {r6.epoch} Training Session Started.', end='')

    model.train()
    scores = []
    for idx, batch in enumerate(dataloader):
        _, targets, outputs, loss = modules.update(batch, model, optimizer, criterion, device=device)
        score = evaluator(outputs, targets)
        scores.append(score.item())
        r6.writer.add_scalar('training loss', loss.item(), global_step=r6.epoch * len(dataloader) + idx)
        r6.writer.add_scalar('train angular error', score.item(), global_step=r6.epoch * len(dataloader) + idx)
        print(f'\rEpoch {r6.epoch} Training Session Proceeding: {idx + 1}/{len(dataloader)}', end='')
    r6.writer.add_scalar('training angular error / epoch', np.nanmean(scores), global_step=r6.epoch)


def valid(dataloader, model, criterion, evaluator, r6):

    print(f'\rEpoch {r6.epoch} Validation Session Started.', end='')

    model.eval()
    scores = []
    for idx, batch in enumerate(dataloader):
        loss, score = modules.evaluate(batch, model, criterion, evaluator, device=device)
        scores.append(score.item())
        r6.writer.add_scalar('validation loss', loss.item(), global_step=r6.epoch * len(dataloader) + idx)
        r6.writer.add_scalar('validation angular error', score.item(), global_step=r6.epoch * len(dataloader) + idx)
        r6.batch_score_board.append(loss.item())
        print(f'\rEpoch {r6.epoch} Validation Session Proceeding: {idx + 1}/{len(dataloader)}', end='')
    r6.writer.add_scalar('validation angular error / epoch', np.nanmean(scores), global_step=r6.epoch)
    print(f'\rEpoch {r6.epoch} Complete.')
    r6.end_epoch()


if __name__ == '__main__':
    main()
