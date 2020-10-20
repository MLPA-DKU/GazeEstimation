import os.path
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms

import config
import datasets
import models
import modules
import utils


def main(args):

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    trainset = args.initialize_object('dataset', datasets, train=True, transform=transform)
    validset = args.initialize_object('dataset', datasets, train=False, transform=transform)
    trainloader = args.initialize_object('loader', loader, trainset, shuffle=True)
    validloader = args.initialize_object('loader', loader, validset, shuffle=False)

    model = args.initialize_object('model', models)
    model.to(args.device)

    criterion = args.initialize_object('criterion', nn)
    evaluator = args.initialize_object('evaluator', modules)
    optimizer = args.initialize_object('optimizer', optim, model.parameters())
    scheduler = args.initialize_object('scheduler', optim.lr_scheduler, optimizer)

    args.writer.args.log_dir = os.path.join(args.log_dir, args.settings.type)
    writer = args.initialize_object('writer', tensorboard)
    args.writer = writer
    checkpoint = utils.Checkpointer('/workspace/save', args.save)

    for epoch in range(args.epochs):
        args.epoch = epoch
        train(trainloader, model, criterion, evaluator, optimizer, args)
        score = validate(validloader, model, criterion, evaluator, args)
        scheduler.step(score)
        checkpoint(model, score)

    args.writer.close()


def train(dataloader, model, criterion, evaluator, optimizer, args):
    dataloader = tqdm.tqdm(dataloader, ncols=200)
    res = utils.ResCapture(dataloader, args, 'train')

    model.train()
    for i, (data, targets) in enumerate(dataloader):
        data, targets = data.to(args.device), targets.to(args.device)

        outputs = model(data)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        args.idx = i
        res(loss.item(), score.item())


def validate(dataloader, model, criterion, evaluator, args):
    dataloader = tqdm.tqdm(dataloader, ncols=200)
    res = utils.ResCapture(dataloader, args, 'valid')

    model.eval()
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(args.device), targets.to(args.device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            score = evaluator(outputs, targets)

            args.idx = i
            res(loss.item(), score.item())
    return np.nanmean(res.results()[1])


if __name__ == '__main__':
    main(config.ConfigParser('config/gaze360.json'))
