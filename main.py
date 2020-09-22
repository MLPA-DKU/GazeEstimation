import os
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dataloader
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms

import datasets.rtgene as rtgene
import models.rtgene.model as models
import modules.modules as mm
import utils

trainlist = ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's010', 's011', 's012', 's013']
validlist = ['s014', 's015', 's016']
inferlist = ['s000']


def main(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    data_type = ['left', 'right']

    trainset = rtgene.RTGENE(root=args.root, transform=transform, subjects=trainlist, data_type=data_type)
    validset = rtgene.RTGENE(root=args.root, transform=transform, subjects=validlist, data_type=data_type)
    trainloader = dataloader.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validloader = dataloader.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = models.GazeEstimator(pretrained=True, device=args.device)
    model = nn.DataParallel(model).to(args.device)

    criterion = nn.MSELoss()
    evaluator = mm.AngleAccuracy()
    optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    writer = tensorboard.SummaryWriter(os.path.join(args.logs, 'rt-gene-vgg16'))

    for epoch in range(args.epochs):
        args.epoch = epoch
        train(trainloader, model, criterion, evaluator, optimizer, writer, args)
        validate(validloader, model, criterion, evaluator, writer, args)
        scheduler.step()
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

    writer.close()


def train(dataloader, model, criterion, evaluator, optimizer, writer, args):

    model.train()
    for i, batch in enumerate(dataloader):
        left, right, headpose, targets = batch
        left, right, headpose = left.to(args.device), right.to(args.device), headpose.to(args.device)
        targets = targets.to(args.device)

        outputs = model(left, right, headpose)
        loss = criterion(outputs, targets)
        accuracy = evaluator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('training loss', loss.item(), args.epoch * len(dataloader) + i)
        writer.add_scalar('training accuracy', accuracy.item(), args.epoch * len(dataloader) + i)

        print(f'Epoch[{args.epoch + 1:4d}/{args.epochs:4d}] - batch[{i + 1:4d}/{len(dataloader):4d}]'
              f' - loss: {loss.item():7.3f} - accuracy: {accuracy.item():7.3f}')


def validate(dataloader, model, criterion, evaluator, writer, args):

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            left, right, headpose, targets = batch
            left, right, headpose = left.to(args.device), right.to(args.device), headpose.to(args.device)
            targets = targets.to(args.device)

            outputs = model(left, right, headpose)
            loss = criterion(outputs, targets)
            accuracy = evaluator(outputs, targets)

            writer.add_scalar('validation loss', loss.item(), args.epoch * len(dataloader) + i)
            writer.add_scalar('validation accuracy', accuracy.item(), args.epoch * len(dataloader) + i)

            print(f'Epoch[{args.epoch + 1:4d}/{args.epochs:4d}] - batch[{i + 1:4d}/{len(dataloader):4d}]'
                  f' - loss: {loss.item():7.3f} - accuracy: {accuracy.item():7.3f}')


if __name__ == '__main__':
    main(utils.ConfigParser('config.json'))
