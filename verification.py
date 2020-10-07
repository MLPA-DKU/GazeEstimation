from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import utils.visualization.cam as viz
import utils.helpers as helpers
from collections import OrderedDict

trainlist = ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's010', 's011', 's012', 's013']
validlist = ['s014', 's015']
inferlist = ['s016']


def main():

    device = 'cuda:1'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    model = torch.load('model_best.pth.tar')
    model.to(device)
    model.eval()

    visualize_layer4 = viz.GradCam(model=model, feature_module=model.layer4, target_layer_names=['1'])

    sample = Image.open('models/example.jpg').convert('RGB')
    transformed_sample = transform(sample).unsqueeze(0).to(device)

    activation_maps_layer4_th = visualize_layer4(transformed_sample, index=0)
    activation_maps_layer4_pi = visualize_layer4(transformed_sample, index=1)
    mask = activation_maps_layer4_th + activation_maps_layer4_pi
    mask = mask - torch.min(mask)
    mask = mask / torch.max(mask)

    helpers.view(sample)
    viz.view_activation_map(sample, mask.squeeze(0).detach().cpu().numpy())


if __name__ == '__main__':
    main()
