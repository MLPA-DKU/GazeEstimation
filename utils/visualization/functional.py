from PIL import Image
import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def visualize_filters(model, index=None, verbose=0):
    modules = list(model.children())
    kernel_weights = []
    index = index if index is not None else 0

    counter = 0
    for module in modules:
        if isinstance(module, nn.Conv2d):
            counter += 1
            kernel_weights.append(module.weight)
        elif isinstance(module, nn.Sequential):
            for i in range(len(module)):
                for child in module[i].children():
                    if isinstance(child, nn.Conv2d):
                        counter += 1
                        kernel_weights.append(child.weight)
    if verbose > 0:
        print(f'total convolutional layers: {counter}')

    fig = plt.figure(figsize=(30, 30))
    for i, kernel in enumerate(kernel_weights[index]):
        out_channels = math.ceil(math.sqrt(kernel_weights[index].shape[0]))
        plt.subplot(out_channels, out_channels, i + 1)
        plt.imshow(kernel[0, :, :].detach(), cmap='jet')
        plt.axis('off')
    plt.show()
    plt.close(fig)
    gc.collect()


def visualize_feature_maps(model, tensor, index=None, verbose=0):
    modules = list(model.children())
    conv_layers = []

    counter = 0
    for module in modules:
        if isinstance(module, nn.Conv2d):
            counter += 1
            conv_layers.append(module)
        elif isinstance(module, nn.Sequential):
            for i in range(len(module)):
                for child in module[i].children():
                    if isinstance(child, nn.Conv2d):
                        counter += 1
                        conv_layers.append(child)
    if verbose > 0:
        print(f'total convolutional layers: {counter}')

    feature_maps = [conv_layers[0](tensor)]
    for i in range(1, len(conv_layers)):
        feature_maps.append(conv_layers[i](feature_maps[-1]))

    index = [int(index)] if index is not None else np.arange(len(feature_maps)).tolist()
    for num_layer in index:
        fig = plt.figure(figsize=(30, 30))
        if verbose > 0:
            print(f'loading layer {num_layer:2d} feature maps...')
        layer_viz = feature_maps[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        for i, feature_map in enumerate(layer_viz):
            if i == 64:
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(feature_map, cmap='jet')
            plt.axis('off')
        plt.show()
        plt.close(fig)
        gc.collect()


def get_gradient_class_activation_maps(model, batch, index, feature_module, model_extractor):
    batch.requires_grad = True

    ret = []
    for image in batch:
        image = image.unsqueeze(0)
        features, output = model_extractor(image)
        index = torch.argmax(output).item() if index is None else index

        one_hot = torch.zeros((1, output.size()[-1]))
        one_hot[0][index] = 1
        one_hot.requires_grad = True
        one_hot = torch.sum(one_hot.to(output.device) * output)

        feature_module.zero_grad()
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        target = features[-1].squeeze()
        weights = nn.AdaptiveAvgPool2d(1)(model_extractor.get_gradients()[-1]).squeeze()

        mask = torch.zeros(target.shape[1:]).to(weights.device)
        for i, w in enumerate(weights):
            mask += w * target[i, :, :]

        mask = mask.clamp(0, 1).view(1, 1, mask.shape[0], mask.shape[1])
        mask = F.interpolate(mask, size=image.shape[2:], mode='bicubic', align_corners=False).squeeze()
        mask = mask - torch.min(mask)
        mask = mask / torch.max(mask)
        ret.append(mask)
    ret = torch.stack(ret)
    return ret


def view_gradient_class_activation_maps(batch, mask):
    heatmap = np.array([plt.get_cmap('jet')(m)[:, :, :3].astype(np.float32) for m in mask])
    batch = batch.numpy().transpose((0, 2, 3, 1)) + heatmap

    for image in batch:
        image = image - image.min()
        image = image / image.max()
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    plt.close()
    gc.collect()


def save_gradient_class_activation_maps(batch, mask, prefix='visualized_activation_map'):
    heatmap = np.array([plt.get_cmap('jet')(m)[:, :, :3].astype(np.float32) for m in mask])
    batch = batch.numpy().transpose((0, 2, 3, 1)) + heatmap

    for i, image in enumerate(batch):
        image = image - image.min()
        image = image / image.max()
        image = image * 255
        image = Image.fromarray(image.astype(np.uint8))
        image.save(f'{prefix}_{i}.png')
