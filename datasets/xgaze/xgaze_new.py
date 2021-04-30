import os
import json
import h5py

from torchvision.datasets import VisionDataset


class XGaze(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(XGaze, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.subjects = 'train' if self.train else 'test'
        self.hdf = None
        self.hdfs = {}

        self.data = []
        self.target = []

        with open(os.path.join(root, 'train_test_split.json'), 'r') as f:
            data_list = json.load(f)

        self.selected_keys = [k for k in data_list[self.subjects]]
        assert len(self.selected_keys) > 0

        for idx in range(len(self.selected_keys)):
            path = os.path.join(self.root, self.subjects, self.selected_keys[idx])
            self.hdfs[idx] = h5py.File(path, 'r', swmr=True)
            assert self.hdfs[idx].swmr_mode

        for idx in range(len(self.selected_keys)):
            num = self.hdfs[idx]['face_patch'].shape[0]
            self.data += [(idx, i) for i in range(num)]

        for idx in range(len(self.hdfs)):
            if self.hdfs[idx]:
                self.hdfs[idx].close()
                self.hdfs[idx] = None

    def __getitem__(self, index):
        key, index = self.data[index]

        self.hdf = h5py.File(os.path.join(self.root, self.subjects, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        inputs = self.hdf['face_patch'][index, :]
        inputs = inputs[:, :, [2, 1, 0]]

        target = self.hdf['face_gaze'][index, :]
        target = target.astype('float')

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return inputs, target

    def __len__(self):
        return len(self.data)
