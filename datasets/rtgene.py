from PIL import Image
import os
import numpy as np

from torchvision.datasets import VisionDataset


class RTGENE(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, subjects=None):
        super(RTGENE, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data = []
        self.headpose = []
        self.gaze = []

        for subject in subjects:
            subject_folder = os.path.join(self.root, f'{subject}_glasses')
            with open(os.path.join(subject_folder, 'label_combined.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    split = line.split(',')
                    face_image = os.path.join(subject_folder, 'inpainted', 'face', f'face_{int(split[0]):06d}_rgb.png')
                    if os.path.exists(face_image):
                        headpose = (float(split[1].strip()[1:]), float(split[2].strip()[:-1]))
                        gaze = (float(split[3].strip()[1:]), float(split[4].strip()[:-1]))
                    self.data.append(face_image)
                    self.headpose.append(headpose)
                    self.gaze.append(gaze)

        self.headpose = np.vstack(self.headpose).astype(np.float32)
        self.gaze = np.vstack(self.gaze).astype(np.float32)

    def __getitem__(self, index):
        image, headpose, target = self.data[index], self.headpose[index], self.gaze[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = np.array(Image.open(image).convert('RGB'))

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, headpose, target

    def __len__(self):
        return len(self.data)


# if __name__ == '__main__':
#     import torch.utils.data
#     import torchvision.transforms as transforms
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     dataset = RTGENE('/Users/winterchild/Downloads/RT-GENE', transform=transform, subjects=['s000', 's001'])
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
#
#     for i, batch in enumerate(dataloader):
#         face, headpose, gaze = batch
#
#     print('')
