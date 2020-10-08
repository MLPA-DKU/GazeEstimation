import os
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import torch.utils.tensorboard as tensorboard
import torchvision.models as models
import torchvision.transforms as transforms

import datasets
import modules
import config
import utils


def main(args):

    best_score = np.inf

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    trainset = datasets.RTGENE(root=args.root, transform=transform, subjects=args.trainlist, data_type=args.data_type)
    validset = datasets.RTGENE(root=args.root, transform=transform, subjects=args.validlist, data_type=args.data_type)
    trainloader = loader.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validloader = loader.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
    model.to(args.device)

    criterion = nn.MSELoss()
    evaluator = modules.AngleError()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=3)

    writer = tensorboard.SummaryWriter(os.path.join(args.logs, 'temp1'))

    for epoch in range(args.epochs):
        args.epoch = epoch
        train(trainloader, model, criterion, evaluator, optimizer, writer, args)
        score = validate(validloader, model, criterion, evaluator, writer, args)
        scheduler.step(score)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

        is_best = score < best_score
        best_score = min(score, best_score)
        utils.save_checkpoint(model, is_best, args.save)

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

            writer.add_scalar('validation loss', loss.item(), args.epoch * len(dataloader) + i)
            writer.add_scalar('validation score', score.item(), args.epoch * len(dataloader) + i)

            print(f'Epoch[{args.epoch + 1:4d}/{args.epochs:4d}] - batch[{i + 1:4d}/{len(dataloader):4d}]'
                  f' - loss: {loss.item():7.3f} - accuracy: {score.item():7.3f}')

    return np.nanmean(res)


if __name__ == '__main__':
    main(config.ConfigParser('config/config.json'))
