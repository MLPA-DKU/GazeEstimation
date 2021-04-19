import torch
from PIL import Image
import os
import json
import h5py
import numpy as np

from torchvision.datasets import VisionDataset


class XGaze(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(XGaze, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            subject_category = 'train'
        else:
            subject_category = 'test'

        self.data = []
        self.target = []

        self.hdf = None
        self.hdfs = {}

        with open(os.path.join(root, 'train_test_split.json'), 'r') as f:
            data_list = json.load(f)

        self.selected_keys = [k for k in data_list[subject_category]]
        assert len(self.selected_keys) > 0

        for idx in range(len(self.selected_keys)):
            path = os.path.join(self.root, subject_category, self.selected_keys[idx])
            self.hdfs[idx] = h5py.File(path, 'r', swmr=True)
            assert self.hdfs[idx].swmr_mode

        for idx in range(len(self.selected_keys)):
            num = self.hdfs[idx]['face_patch'].shape[0]
            self.data += [(num, i) for i in range(num)]

        for idx in range(len(self.hdfs)):
            if self.hdfs[idx]:
                self.hdfs[idx].close()
                self.hdfs[idx] = None

    def __getitem__(self, index):
        data = self.data[index]

        self.hdf = h5py.File(data, 'r', swmr=True)
        assert self.hdf.swmr_mode

        image = self.hdf['face_patch'][index, :]
        image = image[:, :, [2, 1, 0]]

        target = self.hdf['face_gaze'][index, :]
        target = target.astype('float')

        if self.transform is not None:
            data = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import torch.utils.data
    trainset = XGaze('/mnt/saves/ETH-XGaze/xgaze_224', train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=16)
    breakpoint()
