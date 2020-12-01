from PIL import Image
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

random_seed = 42

torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

image = Image.open('example.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
tensor = transform(image)

model = models.resnet152(pretrained=True)
# conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
conv = model.conv1

feature_map = conv(tensor.unsqueeze(0))
feature_map = feature_map.squeeze()

sample = feature_map[2].detach().numpy()
heatmap = plt.get_cmap('jet')(sample)[:, :, :3].astype(np.float32)

fig, axes = plt.subplots(8, 8)

x = 2000 / fig.dpi
y = 2000 / fig.dpi
fig.set_figwidth(x)
fig.set_figheight(y)

for i in range(0, 8):
    for j in range(0, 8):
        sample = feature_map[8 * i + j].detach().numpy()
        heatmap = plt.get_cmap('rainbow')(sample)[:, :, :3].astype(np.float32)
        axes[i][j].imshow(heatmap)

plt.axis('off')
plt.show()

breakpoint()
