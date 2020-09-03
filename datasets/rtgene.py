from PIL import Image
import os
import numpy as np

from torchvision.datasets import VisionDataset


class RTGENE(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, subjects=None, data_type=None):
        super(RTGENE, self).__init__(root, transform=transform, target_transform=target_transform)

        self.face = []
        self.left = []
        self.right = []
        self.headpose = []
        self.gaze = []

        for subject in subjects:
            subject_folder = os.path.join(self.root, f'{subject}_glasses')
            with open(os.path.join(subject_folder, 'label_combined.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    split = line.split(',')
                    face = os.path.join(subject_folder, 'inpainted', 'face', f'face_{int(split[0]):06d}_rgb.png')
                    left = os.path.join(subject_folder, 'inpainted', 'left', f'left_{int(split[0]):06d}_rgb.png')
                    right = os.path.join(subject_folder, 'inpainted', 'right', f'right_{int(split[0]):06d}_rgb.png')
                    if os.path.exists(face) and os.path.exists(left) and os.path.exists(right):
                        headpose = (float(split[1].strip()[1:]), float(split[2].strip()[:-1]))
                        gaze = (float(split[3].strip()[1:]), float(split[4].strip()[:-1]))
                    self.face.append(face)
                    self.left.append(left)
                    self.right.append(right)
                    self.headpose.append(headpose)
                    self.gaze.append(gaze)

        self.headpose = np.vstack(self.headpose).astype(np.float32)
        self.gaze = np.vstack(self.gaze).astype(np.float32)

        self.data_type = ['face', 'left', 'right', 'headpose', 'gaze'] if data_type is None else data_type

    def __getitem__(self, index):

        face = self.face[index]
        left = self.left[index]
        right = self.right[index]
        headpose = self.headpose[index]
        gaze = self.gaze[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        face = np.array(Image.open(face).convert('RGB'))
        left = np.array(Image.open(left).convert('RGB'))
        right = np.array(Image.open(right).convert('RGB'))

        if self.transform is not None:
            face = self.transform(face)
            left = self.transform(left)
            right = self.transform(right)

        if self.target_transform is not None:
            headpose = self.target_transform(headpose)
            gaze = self.target_transform(gaze)

        data = {'face': face, 'left': left, 'right': right, 'headpose': headpose, 'gaze': gaze}
        batch = tuple([data[k] for k in self.data_type])

        return batch

    def __len__(self):
        return len(self.gaze)
