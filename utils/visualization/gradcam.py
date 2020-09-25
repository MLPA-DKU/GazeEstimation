from PIL import Image
import numpy as np
import torch
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
        for name, module in self.model._modules.items():
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
        for name, module in self.model._modules.items():
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

    def __call__(self, image, index=None):

        features, output = self.model_extractor(image)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.model_extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        mask = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            mask += w * target[i, :, :]

        mask = np.maximum(mask, 0)
        mask = Image.fromarray(mask).resize(image.shape[2:])
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        return mask


def view_grad_cam(image, mask, filename='GradCAM.png'):
    image = np.array(image, dtype=np.float32) / 255
    heatmap = plt.get_cmap('jet')(mask)[:, :, :3].astype(np.float32)

    visualize = heatmap + image
    visualize = visualize / np.max(visualize)
    plt.imshow(visualize)
    plt.axis('off')
    plt.show()

    visualize = visualize * 255
    visualize = Image.fromarray(visualize.astype(np.uint8))
    visualize.save(filename)


if __name__ == '__main__':

    import torchvision.models as models
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    model = models.resnet50(pretrained=True)
    grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"])

    image = Image.open('example.jpg').convert('RGB').resize((224, 224))
    transformed_image = transform(image).unsqueeze(0).requires_grad_(True)

    # if target_index = None, returns the map for the highest scoring category.
    # otherwise, targets requires index.
    target_index = None
    mask = grad_cam(transformed_image, target_index)

    view_grad_cam(image, mask)
