import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as loader
import torchvision.transforms as transforms

import callbacks
import datasets
import models
import modules
import modules.optimizer as optim
import utils


torch.autograd.set_detect_anomaly(True)

# global settings
device = 'cuda:0'
epochs = 1000

# dataset option
root = '/mnt/datasets/RT-GENE'
fold_dict = {
    'fold_1': ['s001', 's002', 's008', 's010'],  # fold 1 for train, test
    'fold_2': ['s003', 's004', 's007', 's009'],  # fold 2 for train, test
    'fold_3': ['s005', 's006', 's011', 's012', 's013'],  # fold 3 for train, test
    'fold_4': ['s014', 's015', 's016'],  # fold 4 for validation`
}
subjects_list_train = [
    fold_dict['fold_1'] + fold_dict['fold_2'],  # 1, 2
    fold_dict['fold_1'] + fold_dict['fold_3'],  # 1, 3
    fold_dict['fold_2'] + fold_dict['fold_3'],  # 2, 3
]
subjects_list_valid = [
    fold_dict['fold_4'],
    fold_dict['fold_4'],
    fold_dict['fold_4'],
]
subjects_list_tests = [
    fold_dict['fold_3'],  # 3
    fold_dict['fold_2'],  # 2
    fold_dict['fold_1'],  # 1
]
data_type = ['face']

# dataloader option
batch_size = 128
num_workers = 16


def main():

    for idx, (subjects_train, subjects_valid) in enumerate(zip(subjects_list_train, subjects_list_valid)):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        trainset = datasets.RTGENE(root=root, transform=transform, subjects=subjects_train, data_type=data_type)
        validset = datasets.RTGENE(root=root, transform=transform, subjects=subjects_valid, data_type=data_type)
        trainloader = loader.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validloader = loader.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = models.EEGE()
        model = nn.DataParallel(model)
        model.to(device)

        optimizer = optim.Lookahead(optim.RAdam(model.parameters()))
        criterion = nn.MSELoss()
        evaluator = modules.AngularError()
        checkpoint = callbacks.CheckPoint(directory=f'/tmp/pycharm_project_717/saves/fold_{idx + 1}')
        early_stop = callbacks.EarlyStopping(patience=30)
        best_score = np.inf

        for epoch in range(epochs):
            train(trainloader, model, optimizer, criterion, evaluator, epoch)
            score = valid(validloader, model, criterion, evaluator, epoch)

            early_stop(score)
            if early_stop.early_stop:
                break
            print(f'[ RES ] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
                  f'early stopping count: {early_stop.counter:>{len(str(early_stop.patience))}}/{early_stop.patience}')

            is_best = score < best_score
            best_score = min(score, best_score)
            checkpoint(model, is_best, filename=f'model_epoch_{epoch:>0{len(str(epochs))}d}.pth')


def train(dataloader, model, optimizer, criterion, evaluator, epoch):

    losses = []
    scores = []

    model.train()
    for idx, data in enumerate(dataloader):
        inputs, targets = utils.load_batch(data, device=device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)
        losses.append(loss.item())
        scores.append(score.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'\r[TRAIN] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
              f'batch[{idx + 1:>{len(str(len(dataloader)))}}/{len(dataloader)}] - '
              f'loss: {np.nanmean(losses):.3f} - angular error: {np.nanmean(scores):.3f}', end='')
    print()
    utils.salvage_memory()


def valid(dataloader, model, criterion, evaluator, epoch):

    losses = []
    scores = []

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, targets = utils.load_batch(data, device=device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            score = evaluator(outputs, targets)
            losses.append(loss.item())
            scores.append(score.item())

            print(f'\r[VALID] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
                  f'batch[{idx + 1:>{len(str(len(dataloader)))}}/{len(dataloader)}] - '
                  f'loss: {np.nanmean(losses):.3f} - angular error: {np.nanmean(scores):.3f}', end='')
    print(f'\n[ RES ] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
          f'angular error (°) [{np.nanmean(scores):.3f}|{np.nanstd(scores):.3f}|'
          f'{np.min(scores):.3f}|{np.max(scores):.3f}:MEAN|STD|MIN|MAX]')
    utils.salvage_memory()

    return np.nanmean(scores)


if __name__ == '__main__':
    main()
