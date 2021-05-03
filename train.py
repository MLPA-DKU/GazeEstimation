import torch.nn as nn
import torch.utils.data as loader
import torchvision.transforms as transforms

import datasets
import models
import modules
import utils

utils.enable_easy_debug(False)
utils.enable_reproducibility(False)

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

    trainset = datasets.XGaze(root=root, train=True, transform=transform)
    validset = datasets.XGaze(root=root, train=False, transform=transform)
    trainloader = loader.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = loader.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = models.EEGE()
    model.to(device)

    optimizer = modules.Lookahead(modules.RAdam(model.parameters()))
    criterion = nn.MSELoss()
    evaluator = modules.AngularError()

    # TODO: replace with integrated callback module

    for epoch in range(epochs):
        train(trainloader, model, optimizer, criterion, evaluator)
        valid(validloader, model, criterion, evaluator)


def train(dataloader, model, optimizer, criterion, evaluator):

    model.train()
    for idx, batch in enumerate(dataloader):
        _, targets, outputs, loss = modules.update(batch, model, optimizer, criterion, device=device)
        score = evaluator(outputs, targets)
    utils.salvage_memory()


def valid(dataloader, model, criterion, evaluator):

    model.eval()
    for idx, batch in enumerate(dataloader):
        loss, score = modules.evaluate(batch, model, criterion, evaluator, device=device)
    utils.salvage_memory()


if __name__ == '__main__':
    main()
