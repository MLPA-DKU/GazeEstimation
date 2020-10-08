import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import torch.utils.tensorboard as tensorboard
import torchvision.models as models
import torchvision.transforms as transforms

import config
import datasets
import modules


def main(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    trainset = args.initialize_object('dataset', datasets, transform=transform, subjects=args.trainlist)
    validset = args.initialize_object('dataset', datasets, transform=transform, subjects=args.validlist)
    trainloader = args.initialize_object('loader', loader, trainset, shuffle=True)
    validloader = args.initialize_object('loader', loader, validset, shuffle=False)

    model = args.initialize_object('model', models)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
    model.to(args.device)

    criterion = args.initialize_object('criterion', nn)
    evaluator = args.initialize_object('evaluator', modules)
    optimizer = args.initialize_object('optimizer', optim, model.parameters())
    scheduler = args.initialize_object('scheduler', optim.lr_scheduler, optimizer)

    writer = args.initialize_object('writer', tensorboard)

    for epoch in range(args.epochs):
        args.epoch = epoch
        train(trainloader, model, criterion, evaluator, optimizer, writer, args)
        score = validate(validloader, model, criterion, evaluator, writer, args)
        scheduler.step(score)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

    writer.close()


def train(dataloader, model, criterion, evaluator, optimizer, writer, args):
    model.train()
    for i, batch in enumerate(dataloader):
        face, _, gaze = batch
        face, gaze = face.to(args.device), gaze.to(args.device)

        outputs = model(face)
        loss = criterion(outputs, gaze)
        score = evaluator(outputs, gaze)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer is not None:
            writer.add_scalar('training loss', loss.item(), args.epoch * len(dataloader) + i)
            writer.add_scalar('training score', score.item(), args.epoch * len(dataloader) + i)

        print(f'Epoch[{args.epoch + 1:4d}/{args.epochs:4d}] - batch[{i + 1:4d}/{len(dataloader):4d}]'
              f' - loss: {loss.item():7.3f} - accuracy: {score.item():7.3f}')


def validate(dataloader, model, criterion, evaluator, writer, args):
    res = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            face, _, gaze = batch
            face, gaze = face.to(args.device), gaze.to(args.device)

            outputs = model(face)
            loss = criterion(outputs, gaze)
            score = evaluator(outputs, gaze)

            res.append(loss.item())
            res.append(score.item())

            if writer is not None:
                writer.add_scalar('validation loss', loss.item(), args.epoch * len(dataloader) + i)
                writer.add_scalar('validation score', score.item(), args.epoch * len(dataloader) + i)

            print(f'Epoch[{args.epoch + 1:4d}/{args.epochs:4d}] - batch[{i + 1:4d}/{len(dataloader):4d}]'
                  f' - loss: {loss.item():7.3f} - accuracy: {score.item():7.3f}')

    return np.nanmean(res)


if __name__ == '__main__':
    main(config.ConfigParser('config/config.json'))
