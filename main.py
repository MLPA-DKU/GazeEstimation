import os
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dataloader
import torch.utils.tensorboard as tensorboard
import torchvision.models as models
import torchvision.transforms as transforms


import datasets.rtgene as rtgene
import modules.modules as mm
import utils

trainlist = ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's010', 's011', 's012', 's013']
validlist = ['s014', 's015', 's016']


def main(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    trainset = rtgene.RTGENE(root=args.root, transform=transform, subjects=trainlist)
    validset = rtgene.RTGENE(root=args.root, transform=transform, subjects=validlist)
    trainloader = dataloader.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validloader = dataloader.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = nn.DataParallel(model).to(args.device)


    criterion = nn.MSELoss()
    evaluator = mm.AngleAccuracy()
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    writer = tensorboard.SummaryWriter(os.path.join(args.logs, 'resnet101'))

    for epoch in range(args.epochs):
        args.epoch = epoch
        train(trainloader, model, criterion, evaluator, optimizer, writer, args)
        validate(validloader, model, criterion, evaluator, writer, args)
        scheduler.step()

    writer.close()


def train(dataloader, model, criterion, evaluator, optimizer, writer, args):

    model.train()
    for i, batch in enumerate(dataloader):
        inputs, headpose, targets = batch
        inputs, headpose, targets = inputs.to(args.device), headpose.to(args.device), targets.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accuracy = evaluator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('training loss', loss.item(), args.epoch * len(dataloader) + i)
        writer.add_scalar('training accuracy', accuracy.item(), args.epoch * len(dataloader) + i)

        print(f'Epoch[{args.epoch:4d}/{args.epochs:4d}] - batch[{i:4d}/{len(dataloader):4d}]'
              f' - loss: {loss.item():.3f} - accuracy: {accuracy.item():.3f}')


def validate(dataloader, model, criterion, evaluator, writer, args):

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, headpose, targets = batch
            inputs, headpose, targets = inputs.to(args.device), headpose.to(args.device), targets.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accuracy = evaluator(outputs, targets)

            writer.add_scalar('validation loss', loss.item(), args.epoch * len(dataloader) + i)
            writer.add_scalar('validation accuracy', accuracy.item(), args.epoch * len(dataloader) + i)

            print(f'Epoch[{args.epoch:4d}/{args.epochs:4d}] - batch[{i:4d}/{len(dataloader):4d}]'
                  f' - loss: {loss.item():.3f} - accuracy: {accuracy.item():.3f}')


if __name__ == '__main__':
    main(utils.ConfigParser('config.json'))
