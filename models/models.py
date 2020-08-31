import torch
import torch.nn as nn
import torchvision.models as models

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

print('')
