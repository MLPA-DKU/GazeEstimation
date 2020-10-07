from PIL import Image
import torch
import torchvision.transforms as transforms

import utils.visualization.cam as viz
import utils.helpers as helpers


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
