from PIL import Image
import os
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional

from torchvision.datasets import VisionDataset


class Gaze360(VisionDataset):

    base_folder = 'imgs'
    train_list = list(open('datasets/gaze360/train.txt', 'r'))
    valid_list = list(open('datasets/gaze360/validation.txt', 'r'))
    infer_list = list(open('datasets/gaze360/test.txt', 'r'))

    def __init__(self, root, train=True, transform=None, target_transform=None, mode=None):
        super(Gaze360, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        self.mode = mode if mode is not None else 'video'

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list

        self.data = []
        self.target = []

        for line in downloaded_list:
            line = line[:-1]
            line = line.replace('\t', ' ')
            line = line.replace('  ', ' ')
            split = line.split(' ')
            if len(split) > 3:
                name = int(os.path.split(split[0])[1][:-4])
                path = os.path.join(self.root, self.base_folder, os.path.split(split[0])[0])
                data = [os.path.join(path, f'{name + i:06d}.jpg') for i in range(-3, 4)]
                target = np.array([float(v) for v in split[1:4]])
                self.data.append(data)
                self.target.append(target)

    def __getitem__(self, index):

        data = self.data[index]
        target = self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        data = [Image.open(p).convert('RGB').resize((224, 224)) for p in data]
        data = data if self.mode == 'video' else data[3]

        target = torch.tensor(target, dtype=torch.float32)
        target = nn.functional.normalize(target.view(1, 3)).view(3)
        target = [torch.atan2(target[0], -target[2]), torch.asin(target[1])]
        target = torch.stack(target)

        if self.transform is not None:
            data = torch.stack([self.transform(f) for f in data]) if self.mode == 'video' else self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        data = torch.cat([f for f in data], dim=0) if self.mode == 'video' else data

        return data, target

    def __len__(self):
        return len(self.data)


class Gaze360Inference(Gaze360):

    def __init__(self, root, transform=None, target_transform=None, mode=None):
        super(Gaze360Inference, self).__init__(root, transform=transform, target_transform=target_transform, mode=mode)

        self.mode = mode if mode is not None else 'video'

        downloaded_list = self.infer_list

        self.data = []
        self.target = []

        for line in downloaded_list:
            line = line[:-1]
            line = line.replace('\t', ' ')
            line = line.replace('  ', ' ')
            split = line.split(' ')
            if len(split) > 3:
                name = int(os.path.split(split[0])[1][:-4])
                path = os.path.join(self.root, self.base_folder, os.path.split(split[0])[0])
                data = [os.path.join(path, f'{name + i:06d}.jpg') for i in range(-3, 4)]
                target = np.array([float(v) for v in split[1:4]])
                self.data.append(data)
                self.target.append(target)
