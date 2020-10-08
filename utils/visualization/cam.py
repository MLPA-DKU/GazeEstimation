from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FeatureExtractor:

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in dict(self.model.named_children()).items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelFeatureExtractor:

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in dict(self.model.named_children()).items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        return target_activations, x


class GradCam:

    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
        self.model.eval()
        self.feature_module = feature_module
        self.model_extractor = ModelFeatureExtractor(self.model, self.feature_module, target_layer_names)

    def forward(self, x):
        return self.model(x)

    def __call__(self, batch, index=None):
        batch.requires_grad = True

        ret = []
        for image in batch:
            image = image.unsqueeze(0)
            features, output = self.model_extractor(image)
            index = torch.argmax(output).item() if index is None else index

            one_hot = torch.zeros((1, output.size()[-1]))
            one_hot[0][index] = 1
            one_hot.requires_grad = True
            one_hot = torch.sum(one_hot.to(output.device) * output)

            self.feature_module.zero_grad()
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            target = features[-1].squeeze()
            weights = nn.AdaptiveAvgPool2d(1)(self.model_extractor.get_gradients()[-1]).squeeze()

            mask = torch.zeros(target.shape[1:]).to(weights.device)
            for i, w in enumerate(weights):
                mask += w * target[i, :, :]

            mask = mask.clamp(0, 1).view(1, 1, mask.shape[0], mask.shape[1])
            mask = F.interpolate(mask, size=image.shape[2:], mode='bicubic', align_corners=False).squeeze()
            mask = mask - torch.min(mask)
            mask = mask / torch.max(mask)
            ret.append(mask)

        return torch.stack(ret)


def view_activation_map(image, mask):
    image = np.array(image, dtype=np.float32) / 255
    heatmap = plt.get_cmap('jet')(mask)[:, :, :3].astype(np.float32)

    visualize = heatmap + image
    visualize = visualize / np.max(visualize)
    plt.imshow(visualize)
    plt.axis('off')
    plt.show()


def save_activation_map(image, mask, filename='visualized_activation_map.png'):
    image = np.array(image, dtype=np.float32) / 255
    heatmap = plt.get_cmap('jet')(mask)[:, :, :3].astype(np.float32)

    visualize = heatmap + image
    visualize = visualize / np.max(visualize)
    visualize = visualize * 255
    visualize = Image.fromarray(visualize.astype(np.uint8))
    visualize.save(filename)
