import numpy as np
import torch.nn as nn
import torch.utils.data as loader
import torchvision.transforms as transforms

import datasets
import models
import modules
import modules.optimizers as optim
import utils

utils.enable_easy_debug(False)
utils.enable_reproducibility(False)

# global settings
device = 'cuda:0'
epochs = 1000

# dataset option
root = '/mnt/datasets/Gaze/Gaze360'

# dataloader option
batch_size = 32
num_workers = 16


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(512, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    trainset = datasets.Gaze360(root=root, train=True, transform=transform, mode='frame')
    validset = datasets.Gaze360(root=root, train=False, transform=transform, mode='frame')
    trainloader = loader.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = loader.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = models.EEGE()
    model.to(device)

    optimizer = optim.Lookahead(optim.RAdam(model.parameters()))
    criterion = nn.MSELoss()
    evaluator = modules.AngularError()

    # TODO: replace with integrated callback module

    for epoch in range(epochs):
        train(trainloader, model, optimizer, criterion, evaluator, epoch)
        score = valid(validloader, model, criterion, evaluator, epoch)

def train(dataloader, model, optimizer, criterion, evaluator, epoch):

    losses = []
    scores = []

    model.train()
    for idx, batch in enumerate(dataloader):
        _, targets, outputs, loss = utils.update(batch, model, optimizer, criterion, device=device)
        score = evaluator(outputs, targets)
        losses.append(loss.item())
        scores.append(score.item())
        utils.print_result(epoch, epochs, idx, dataloader, losses, scores, header='TRAIN')
    print()
    utils.salvage_memory()


def valid(dataloader, model, criterion, evaluator, epoch):

    losses = []
    scores = []

    model.eval()
    for idx, batch in enumerate(dataloader):
        loss, score = utils.evaluate(batch, model, criterion, evaluator, device=device)
        losses.append(loss.item())
        scores.append(score.item())
        utils.print_result(epoch, epochs, idx, dataloader, losses, scores, header='VALID')
    utils.print_result_on_epoch_end(epoch, epochs, scores)
    utils.salvage_memory()

    return np.nanmean(scores)


if __name__ == '__main__':
    main()
