import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

feature_blobs = []


def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy())


model = models.resnet50()
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)

model._modules.get('layer4').register_forward_hook(hook_feature)

param = list(model.parameters())
weight_softmax = np.squeeze(param[-2].cpu().data.numpy())

breakpoint()
