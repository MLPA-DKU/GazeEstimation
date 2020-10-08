from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    sample = Image.open('example.jpg').convert('RGB').resize((224, 224))
    transformed_sample = transform(sample).unsqueeze(0)

    model = torch.load('/tmp/pycharm_project_662/resnet18.pth', map_location='cpu')
    model.eval()

    model_weights = []
    conv_layers = []

    model_children = list(model.children())

    counter = 0
    for i in range(len(model_children)):
        if isinstance(model_children[i], nn.Conv2d):
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif isinstance(model_children[i], nn.Sequential):
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if isinstance(child, nn.Conv2d):
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f'total convolutional layers: {counter}')

    # for weight, conv in zip(model_weights, conv_layers):
    #     print(f'CONV: {conv} ==> SHAPE: {weight.shape}')

    # plt.figure(figsize=(20, 17))
    # for i, filter in enumerate(model_weights[0]):
    #     plt.subplot(8, 8, i + 1)
    #     plt.imshow(filter[0, :, :].detach(), cmap='jet')
    #     plt.axis('off')
    # plt.show()

    results = [conv_layers[0](transformed_sample)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))

    outputs = results

    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        print(f'loading layer {num_layer:2d} feature maps...')
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        for i, feature in enumerate(layer_viz):
            if i == 64:
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(feature, cmap='jet')
            plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
