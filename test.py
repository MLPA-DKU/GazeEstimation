import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import datasets
import modules
import modules.engine as engine
import utils

# global settings
device = utils.auto_device()

# dataset option
root = '/mnt/datasets/Gaze/Gaze360'

# dataloader option
num_workers = 8


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_set = datasets.Gaze360Inference(root=root, transform=transform, mode='frame')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)

    model = torch.load(...)
    model.to(device)

    evaluator = modules.AngularError()

    score = test(test_loader, model, evaluator)


def test(dataloader, model, evaluator):

    scores = []

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, targets = engine.load_batch(data, device=device)
            outputs = model(inputs)
            score = evaluator(outputs, targets)
            scores.append(score.item())

            print(f'\rTest Sample {len(dataloader)} | Progress... {int((idx + 1) / len(dataloader) * 100):>3d}%', end='')

    print(f'\nTest Report - '
          f'Mean: {np.nanmean(scores):.3f}, Std: {np.nanstd(scores):.3f}, '
          f'Min:{np.min(scores):.3f}, Max:{np.max(scores):.3f}')

    utils.salvage_memory()

    return scores


if __name__ == '__main__':
    main()
