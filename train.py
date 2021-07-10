import numpy as np
import torch.nn
import torch.utils.data as loader
import torchvision.transforms as transforms

import datasets
import models
import modules
import utils


def create_dataloader_set(root, updater, num_workers=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    trainset = datasets.Gaze360(root=root, train=True, transform=transform, mode='frame')
    validset = datasets.Gaze360(root=root, train=False, transform=transform, mode='frame')
    batch_size = utils.auto_batch_size(trainset, updater)
    trainloader = loader.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = loader.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, validloader


def main() -> None:

    volume_data = '/mnt/datasets/gaze360'
    volume_save = '/mnt/experiments/gaze360'

    utils.setup_logger()
    device = utils.auto_device()

    model = ...
    model.to(device)

    optimizer = modules.Lookahead(modules.RAdam(model.parameters()))
    criterion = torch.nn.MSELoss()
    evaluator = [criterion, modules.AngularError()]

    trainfunction = modules.update(model, optimizer, criterion, device)
    validfunction = modules.evaluate(model, device)

    trainloader, validloader = create_dataloader_set(volume_data, trainfunction)

    board_writer = utils.create_tensorboard_writer(volume_save)
    checkpointer = {
        'save': utils.create_checkpoint_handler({'model': model}, volume_save),
        'perf': utils.create_performance_meter()
    }

    for epoch in range(10000):
        train(epoch, trainloader, trainfunction, evaluator, board_writer)
        valid(epoch, validloader, validfunction, evaluator, board_writer, checkpointer)


def train(epoch, dataloader, process_fn, eval_functions, writer) -> None:
    for idx, batch in enumerate(dataloader):
        outputs = process_fn(batch)
        outputs = [eval(*outputs) for eval in eval_functions]
        writer.log('train_loss', outputs[0], epoch * len(dataloader) + idx)
        writer.log('train_angular_error', outputs[1], epoch * len(dataloader) + idx)


def valid(epoch, dataloader, process_fn, eval_functions, writer, save_functions) -> None:
    perf = []
    for idx, batch in enumerate(dataloader):
        outputs = process_fn(batch)
        outputs = [eval(*outputs) for eval in eval_functions]
        perf.append(outputs[0].item())
        writer.log('valid_loss', outputs[0], epoch * len(dataloader) + idx)
        writer.log('valid_angular_error', outputs[1], epoch * len(dataloader) + idx)
    is_best = save_functions['perf'](np.nanmean(perf))
    if is_best:
        save_functions['save'](f'checkpoint_best_perf.pth')


if __name__ == '__main__':
    main()
