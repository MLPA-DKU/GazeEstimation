import os
import json
import h5py

from torchvision.datasets import VisionDataset


class XGaze(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(XGaze, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.subjects = 'train' if self.train else 'test'

        self.data = []

        with open(os.path.join(root, 'train_test_split.json'), 'r') as f:
            data_list = json.load(f)

        self.selected_keys = [k for k in data_list[self.subjects]]
        assert len(self.selected_keys) > 0

        for idx in range(len(self.selected_keys)):
            hdf = h5py.File(os.path.join(self.root, self.subjects, self.selected_keys[idx]), 'r', swmr=True)
            assert hdf.swmr_mode
            self.data += [(idx, i) for i in range(hdf['face_patch'].shape[0])]
            hdf.close()

    def __getitem__(self, index):
        key, index = self.data[index]

        hdf = h5py.File(os.path.join(self.root, self.subjects, self.selected_keys[key]), 'r', swmr=True)
        assert hdf.swmr_mode

        inputs = hdf['face_patch'][index, :]
        inputs = inputs[:, :, [2, 1, 0]]

        targets = hdf['face_gaze'][index, :]
        targets = targets.astype('float')

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return inputs, targets

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = XGaze(root='/mnt/saves/ETH-XGaze/xgaze_224', train=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    breakpoint()
