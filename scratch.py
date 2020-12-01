import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import datasets


def spherical_to_cartesian(vector):
    x = torch.cos(vector[:, 1]) * torch.sin(vector[:, 0])
    y = torch.sin(vector[:, 1])
    z = -1 * torch.cos(vector[:, 1]) * torch.cos(vector[:, 0])
    return torch.stack((x, y, z), dim=1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
])

trainset = datasets.RTGENE(root='/mnt/datasets/RT-GENE', transform=transform, subjects=['s001'], data_type=['face'])
sample = trainset[6642]

face_image, gaze_vector = sample

face_image = face_image.numpy().transpose((1, 2, 0))

gaze_vector = torch.from_numpy(gaze_vector)
gaze_vector_cartesian = spherical_to_cartesian(gaze_vector.unsqueeze(0))
gaze_vector_cartesian = gaze_vector_cartesian.numpy()

plt.imshow(face_image)
plt.show()

breakpoint()
