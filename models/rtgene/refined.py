import torch
import matplotlib.pyplot as plt


resolution = 224, 224, 3
batch_size = 512
input_size = list()
input_size.append(batch_size)
input_size.extend(resolution)

batch = torch.randint(low=0, high=255, size=input_size)

for i in range(len(batch)):
    image = batch[i].numpy()
    breakpoint()

breakpoint()
