import torch
import torch.nn as nn
import torch.utils.data as loader
import torchvision.models as models
import torchvision.transforms as transforms

import datasets.rtgene as rtgene
import modules.modules as mm
import utils.visualization.cam as viz
import utils.helpers as helpers
from collections import OrderedDict


trainlist = ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's010', 's011', 's012', 's013']
validlist = ['s014', 's015', 's016']
inferlist = ['s000']


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def main():

    device = 'cuda:0'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    undo_normalize = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    data_type = ['face']

    validset = rtgene.RTGENE(root='/mnt/datasets/RT-GENE', transform=transform, subjects=validlist, data_type=data_type)
    validloader = loader.DataLoader(validset, batch_size=1, shuffle=True)

    model = models.resnet152()
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)

    state_dict = torch.load('model.pth').state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.to(device)
    visualize_layer1 = viz.GradCam(model=model, feature_module=model.layer1, target_layer_names=['2'])
    visualize_layer2 = viz.GradCam(model=model, feature_module=model.layer2, target_layer_names=['2'])
    visualize_layer3 = viz.GradCam(model=model, feature_module=model.layer3, target_layer_names=['2'])
    visualize_layer4 = viz.GradCam(model=model, feature_module=model.layer4, target_layer_names=['2'])

    evaluator = mm.AngleAccuracy()

    for _, batch in enumerate(validloader):
        face, _, gaze = batch
        face, gaze = face.to(device), gaze.to(device)

        outputs = model(face)
        activation_maps_layer1_th = visualize_layer1(face, index=0)
        activation_maps_layer1_pi = visualize_layer1(face, index=1)
        activation_maps_layer2_th = visualize_layer2(face, index=0)
        activation_maps_layer2_pi = visualize_layer2(face, index=1)
        activation_maps_layer3_th = visualize_layer3(face, index=0)
        activation_maps_layer3_pi = visualize_layer3(face, index=1)
        activation_maps_layer4_th = visualize_layer4(face, index=0)
        activation_maps_layer4_pi = visualize_layer4(face, index=1)

        for i in range(1):
            accuracy = evaluator(outputs[i].unsqueeze(0), gaze[i].unsqueeze(0))
            print(f'accuracy of image {i}: {accuracy.item():.3f}')

        for i in range(1):
            image = transforms.ToPILImage()(undo_normalize(face[i]).cpu())
            helpers.view(image)
            viz.view_activation_map(image, activation_maps_layer1_th[i].detach().cpu().numpy())
            viz.view_activation_map(image, activation_maps_layer2_th[i].detach().cpu().numpy())
            viz.view_activation_map(image, activation_maps_layer3_th[i].detach().cpu().numpy())
            viz.view_activation_map(image, activation_maps_layer4_th[i].detach().cpu().numpy())
            viz.view_activation_map(image, activation_maps_layer1_pi[i].detach().cpu().numpy())
            viz.view_activation_map(image, activation_maps_layer2_pi[i].detach().cpu().numpy())
            viz.view_activation_map(image, activation_maps_layer3_pi[i].detach().cpu().numpy())
            viz.view_activation_map(image, activation_maps_layer4_pi[i].detach().cpu().numpy())

        breakpoint()

        break


if __name__ == '__main__':
    main()
