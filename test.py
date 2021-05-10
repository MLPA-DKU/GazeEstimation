import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as loader
import torchvision.transforms as transforms

import datasets
import models
import modules
import utils

# global settings
device = utils.auto_device()
epochs = 1000

# dataset option
root = '/mnt/datasets/Gaze/Gaze360'
save = '/mnt/saves/Gaze'

# dataloader option
batch_size = 1
num_workers = 16


def main():

    print('\rInitializing...', end='')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    testset = datasets.Gaze360Inference(root=root, transform=transform, mode='frame')
    testloader = loader.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = torch.load(os.path.join(save, 'checkpoint', 'checkpoint.872495e0', 'checkpoint.R6.872495e0.score.best.pth'))
    model = model['model']
    model.to(device)

    criterion = nn.MSELoss()
    evaluator = modules.AngularError()

    print('\rInitialization Complete.')

    test(testloader, model, criterion, evaluator)


def test(dataloader, model, criterion, evaluator):

    model.eval()
    losses = []
    scores = []
    for idx, batch in enumerate(dataloader):
        loss, score = modules.evaluate(batch, model, criterion, evaluator, device=device)
        losses.append(loss.item())
        scores.append(score.item())
        print(f'\rTest Session Proceeding: {idx + 1}/{len(dataloader)}', end='')
    print(f'\nSummary Report')
    print(f'Test Loss')
    print(f'Mean: {np.nanmean(losses):.3f} | Std: {np.nanstd(losses):.3f} | Min: {np.nanmin(losses):.3f} | Max: {np.nanmax(losses):.3f}')
    print(f'Test Angular Error')
    print(f'Mean: {np.nanmean(scores):.3f} | Std: {np.nanstd(scores):.3f} | Min: {np.nanmin(scores):.3f} | Max: {np.nanmax(scores):.3f}')


if __name__ == '__main__':
    main()
