import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader

import datasets
import models
import modules
import utils


def main(args):

    trainset = args.initialize_object('dataset', datasets, train=True, transform=utils.transform)
    validset = args.initialize_object('dataset', datasets, train=False, transform=utils.transform)
    trainloader = args.initialize_object('loader', loader, trainset, shuffle=True)
    validloader = args.initialize_object('loader', loader, validset, shuffle=False)

    model = args.initialize_object('model', models)
    model.to(args.device)

    criterion = args.initialize_object('criterion', nn)
    evaluator = args.initialize_object('evaluator', modules)
    optimizer = args.initialize_object('optimizer', optim, model.parameters())
    scheduler = args.initialize_object('scheduler', optim.lr_scheduler, optimizer)

    for epoch in range(args.epochs):
        args.epoch = epoch
        train(trainloader, model, criterion, evaluator, optimizer, args)
        score = validate(validloader, model, criterion, evaluator, args)
        scheduler.step(score)


def train(dataloader, model, criterion, evaluator, optimizer, args):
    model.train()
    for i, (data, targets) in enumerate(dataloader):
        data, targets = data.to(args.device), targets.to(args.device)

        outputs = model(data)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(dataloader, model, criterion, evaluator, args):
    scores = []
    model.eval()
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(args.device), targets.to(args.device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            score = evaluator(outputs, targets)
            scores.append(score.item())

    return np.nanmean(scores)


if __name__ == '__main__':
    main(utils.ConfigParser('utils/config/gaze360.json'))
