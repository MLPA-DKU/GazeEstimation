import torch
import torch.nn as nn
import utils.visualization.cam as viz


class SpatialMask(nn.Module):

    def __init__(self, backbone, grad_cam):
        super(SpatialMask, self).__init__()
        self.backbone = backbone
        self.grad_cam = grad_cam

    def forward(self, x):
        _ = self.backbone(x)
        a = self.grad_cam(x)
        return torch.cat((x, a.unsqueeze(1)), dim=1)


if __name__ == '__main__':

    from PIL import Image
    import torchvision.models as models
    import torchvision.transforms as transforms

    device = 'cpu'

    model = models.resnet18()
    visualizer = viz.GradCam(model, model.layer4, ["1"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    image = Image.open('example.jpg').convert('RGB').resize((224, 224))
    transformed_image = transform(image).unsqueeze(0)
    for _ in range(2):
        transformed_image = torch.cat([transformed_image, transformed_image], dim=0)
    transformed_image = transformed_image.to(device)

    spatial_mask = SpatialMask(model, visualizer)
    concatenate_image = spatial_mask(transformed_image)

    breakpoint()
