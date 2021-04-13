from PIL import Image
import os
import numpy as np

from torchvision.datasets import VisionDataset

fold_dict = {
    'fold_1': ['s001', 's002', 's008', 's010'],  # fold 1 for train, test
    'fold_2': ['s003', 's004', 's007', 's009'],  # fold 2 for train, test
    'fold_3': ['s005', 's006', 's011', 's012', 's013'],  # fold 3 for train, test
    'fold_4': ['s014', 's015', 's016'],  # fold 4 for validation
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


class RTGENE(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, subjects=None, data_type=None):
        super(RTGENE, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data = []
        self.headpose = []
        self.target = []

        self.data_type = ['head', 'face', 'left', 'right', 'headpose'] if data_type is None else data_type

        for subject in subjects:
            subject_folder = os.path.join(self.root, f'{subject}_glasses')
            with open(os.path.join(subject_folder, 'label_combined.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    split = line.split(',')
                    base_folder = os.path.join(subject_folder, 'inpainted')
                    head = os.path.join(base_folder, 'face_after_inpainting', f'{int(split[0]):06d}.png')
                    face = os.path.join(base_folder, 'face', f'face_{int(split[0]):06d}_rgb.png')
                    left = os.path.join(base_folder, 'left', f'left_{int(split[0]):06d}_rgb.png')
                    right = os.path.join(base_folder, 'right', f'right_{int(split[0]):06d}_rgb.png')
                    if os.path.exists(head) and os.path.exists(face) and os.path.exists(left) and os.path.exists(right):
                        headpose = (float(split[1].strip()[1:]), float(split[2].strip()[:-1]))
                        gaze = (float(split[3].strip()[1:]), float(split[4].strip()[:-1]))
                    data = {'head': head, 'face': face, 'left': left, 'right': right}
                    data = [data[k] for k in self.data_type]

                    self.data.append(data)
                    self.headpose.append(headpose)
                    self.target.append(gaze)

        self.headpose = np.vstack(self.headpose).astype(np.float32)
        self.target = np.vstack(self.target).astype(np.float32)

    def __getitem__(self, index):
        data = self.data[index]
        headpose = self.headpose[index]
        target = self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        data = [np.array(Image.open(path).convert('RGB')) for path in data]

        if self.transform is not None:
            data = [self.transform(image) for image in data]

        if self.target_transform is not None:
            headpose = self.target_transform(headpose)
            target = self.target_transform(target)

        batch = []
        batch.extend(data)
        batch.append(headpose) if 'headpose' in self.data_type else None
        batch.append(target)

        return batch

    def __len__(self):
        return len(self.data)
