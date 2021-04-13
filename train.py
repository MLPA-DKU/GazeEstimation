import time
import numpy as np
import torch.nn as nn
import torch.utils.data as loader
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms

import datasets
import models
import modules
import modules.optimizers as optim
import profiler
import utils

utils.enable_easy_debug(False)
utils.enable_reproducibility(False)

# global settings
device = 'cuda:0'
epochs = 1000

# dataset option
root = '/mnt/datasets/RT-GENE'
data_type = ['face']

# dataloader option
batch_size = 32
num_workers = 16


def main():

    subjects_list_train = datasets.rtgene.subjects_list_train
    subjects_list_valid = datasets.rtgene.subjects_list_valid

    for idx, (subjects_train, subjects_valid) in enumerate(zip(subjects_list_train, subjects_list_valid)):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(512, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
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

        best_score = np.inf
        checkpoint = profiler.CheckPoint(save_dir=f'/tmp/pycharm_project_717/saves/fold_{idx + 1}')
        early_stop = profiler.EarlyStopping(patience=30, epochs=epochs)
        writer = tensorboard.SummaryWriter(log_dir=f"./logs/{int(str(time.time()).split('.')[1]):07d}")

        for epoch in range(epochs):
            train(trainloader, model, optimizer, criterion, evaluator, writer, epoch)
            score = valid(validloader, model, criterion, evaluator, writer, epoch)

            early_stop(score, epoch)
            if early_stop.early_stop:
                break

            is_best = score < best_score
            best_score = min(score, best_score)
            checkpoint(model, is_best, checkpoint_name=f'model_epoch_{epoch:>0{len(str(epochs))}d}.pth')


def train(dataloader, model, optimizer, criterion, evaluator, writer, epoch):

    losses = []
    scores = []

    model.train()
    for idx, batch in enumerate(dataloader):
        _, targets, outputs, loss = utils.update(batch, model, optimizer, criterion, device=device)
        score = evaluator(outputs, targets)
        losses.append(loss.item())
        scores.append(score.item())

        writer.add_scalar('[TRAIN] Losses', loss.item(), global_step=epoch*len(dataloader)+idx)
        writer.add_scalar('[TRAIN] Scores', score.item(), global_step=epoch*len(dataloader)+idx)
        utils.print_result(epoch, epochs, idx, dataloader, losses, scores, header='TRAIN')
    print()
    utils.salvage_memory()


def valid(dataloader, model, criterion, evaluator, writer, epoch):

    losses = []
    scores = []

    model.eval()
    for idx, batch in enumerate(dataloader):
        loss, score = utils.evaluate(batch, model, criterion, evaluator, device=device)
        losses.append(loss.item())
        scores.append(score.item())

        writer.add_scalar('[VALID] Losses', loss.item(), global_step=epoch*len(dataloader)+idx)
        writer.add_scalar('[VALID] Scores', score.item(), global_step=epoch*len(dataloader)+idx)
        utils.print_result(epoch, epochs, idx, dataloader, losses, scores, header='VALID')
    utils.print_result_on_epoch_end(epoch, epochs, scores)
    utils.salvage_memory()

    return np.nanmean(scores)


if __name__ == '__main__':
    main()
