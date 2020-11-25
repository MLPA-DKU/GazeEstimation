import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import torchvision.transforms as transforms

import datasets
import models
import modules

epochs = 1000


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    trainset = datasets.RTGENE(..., transform=transform, subjects=[], data_type=[])
    validset = datasets.RTGENE(..., transform=transform, subjects=[], data_type=[])
    test_set = datasets.RTGENE(..., transform=transform, subjects=[], data_type=[])
    trainloader = loader.DataLoader(trainset, ...)
    validloader = loader.DataLoader(validset, ...)
    test_loader = loader.DataLoader(test_set, ...)

    model = models.EfficientDet()

    criterion = nn.MSELoss()
    evaluator = modules.AngleError()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

    callbacks = ...

    for epoch in range(epochs):
        train(trainloader, model, optimizer, criterion, evaluator, callbacks)
        validate(validloader, model, criterion, evaluator, callbacks)
        scheduler.step(epoch)

        if callbacks.early_stop:
            break

    test(test_loader, model)


def train(dataloader, model, optimizer, criterion, evaluator, callbacks):
    model.train()
    for idx, data in enumerate(dataloader):
        inputs, targets = data
        inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        callbacks.history()
        callbacks.checkpoint()


def validate(dataloader, model, criterion, evaluator, callbacks):
    model.eval()
    for idx, data in enumerate(dataloader):
        inputs, targets = data
        inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)

        callbacks.history()


def test(dataloader, model):
    model.eval()
    for idx, data in enumerate(dataloader):
        inputs, targets = data
        inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')

        outputs = model(inputs)


if __name__ == '__main__':
    main()
