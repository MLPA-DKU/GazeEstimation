import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
# import torchvision.models as models
import torchvision.transforms as transforms

import config
import datasets
import models
import modules


def main(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    trainset = args.initialize_object('dataset', datasets, train=True, transform=transform)
    validset = args.initialize_object('dataset', datasets, train=False, transform=transform)
    trainloader = args.initialize_object('loader', loader, trainset, shuffle=True)
    validloader = args.initialize_object('loader', loader, validset, shuffle=False)

    model = args.initialize_object('model', models)
    model.to(args.device)

    criterion = args.initialize_object('criterion', modules)
    evaluator = args.initialize_object('evaluator', modules)
    optimizer = args.initialize_object('optimizer', optim, model.parameters())
    scheduler = args.initialize_object('scheduler', optim.lr_scheduler, optimizer)

    for epoch in range(args.epochs):
        args.epoch = epoch
        train(trainloader, model, criterion, evaluator, optimizer, args)
        validate(validloader, model, criterion, evaluator, args)
        scheduler.step()


def train(dataloader, model, criterion, evaluator, optimizer, args):
    model.train()
    for i, (data, targets) in enumerate(dataloader):
        data, targets = data.to(args.device), targets.to(args.device)

        outputs, var = model(data)
        loss = criterion(outputs, targets, var)
        score = evaluator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch[{args.epoch + 1:4d}/{args.epochs:4d}] - batch[{i + 1:4d}/{len(dataloader):4d}]'
              f' - loss: {loss.item():7.3f} - accuracy: {score.item():7.3f}')


def validate(dataloader, model, criterion, evaluator, args):
    model.eval()
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(args.device), targets.to(args.device)

            outputs, var = model(data)
            loss = criterion(outputs, targets, var)
            score = evaluator(outputs, targets)

            print(f'Epoch[{args.epoch + 1:4d}/{args.epochs:4d}] - batch[{i + 1:4d}/{len(dataloader):4d}]'
                  f' - loss: {loss.item():7.3f} - accuracy: {score.item():7.3f}')


if __name__ == '__main__':
    main(config.ConfigParser('config/gaze360.json'))
