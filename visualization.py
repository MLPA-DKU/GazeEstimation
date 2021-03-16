from PIL import ImageDraw
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import datasets
import utils

root = '/mnt/datasets/Gaze360'
device = 'cuda:0'


def visualize_gaze_direction_gaze360(tensor, gaze, prediction=None, save=None, length=200):

    # PyTorch image tensor shape: CHW
    # Input 'tensor' should be 3D tensor (Channel, Height, Width)
    # Input 'gaze' should be 2D vector (theta, phi)

    def transform_coordinate_system(theta, phi, o):
        x = int((-1 * torch.sin(theta) * torch.cos(phi) * length + o[0]).item())
        y = int((-1 * torch.sin(phi) * length + o[1]).item())
        return x, y

    image = transforms.ToPILImage()(tensor.detach())
    image_overlay = ImageDraw.Draw(image)

    endpoint = transform_coordinate_system(gaze[0], gaze[1], (image.width // 2, image.height // 2))
    image_overlay.line([(int(image.width // 2), int(image.height // 2)), endpoint], fill='blue', width=3)

    if prediction:
        predict = transform_coordinate_system(prediction[0], prediction[1], (image.width // 2, image.height // 2))
        image_overlay.line([(int(image.width // 2), int(image.height // 2)), predict], fill='blue', width=3)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    plt.close()

    if save is not None:
        image.save(save)


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(380),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = datasets.Gaze360(root=root, train=True, transform=transform, mode='image')
    dataset = torch.utils.data.Subset(dataset, np.arange(50000, 50100))

    image, target = dataset[90]
    image = utils.denorm(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    visualize_gaze_direction_gaze360(image, target)


if __name__ == '__main__':
    main()
