import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import datasets
import modules
import modules.engine as engine
import utils

# global settings
device = 'cuda:0'

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
num_workers = 8


def main():

    scores = []

    for idx, subjects_tests in enumerate(subjects_list_tests):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        test_set = datasets.RTGENE(root=root, transform=transform, subjects=subjects_tests, data_type=data_type)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)

        model = torch.load(f'/tmp/pycharm_project_182/saves/fold_{idx + 1}/model_best.pth')
        model = nn.DataParallel(model)
        model.to(device)

        evaluator = modules.AngularError()

        score = test(test_loader, model, evaluator)
        scores.extend(score)

    print(f'Mean angular error of 3-fold: {np.nanmean(scores)}')


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
