import torch
import torchvision.models as models

iterations = 20
resolution = 224
batch_size = 12

device = 'cuda:0'

model = models.resnet101().to(device)
model.eval()

inputs = torch.rand((batch_size, 3, resolution, resolution)).to(device)

# for warm up
for i in range(5):
    with torch.no_grad():
        model(inputs)

cpu_time_ms = 0
gpu_time_ms = 0
for i in range(iterations):
    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            model(inputs)
        cpu_time_ms += prof.self_cpu_time_total / 1000
        gpu_time_ms += sum([event.cuda_time_total for event in prof.key_averages()]) / 1000

cpu_time_ms /= iterations * batch_size
gpu_time_ms /= iterations * batch_size

width = 40
print(f'=' * width)
print(f'ResNet101'.center(width))
print(f'Nvidia GeForce GTX Titan Xp'.center(width))
print(f'image resolution: {resolution}x{resolution}, RGB'.center(width))
print(f'inference time (cpu): {cpu_time_ms:7.1f} ms'.center(width))
print(f'inference time (gpu): {gpu_time_ms:7.1f} ms'.center(width))
print(f'inference time (total): {cpu_time_ms + gpu_time_ms:5.1f} ms'.center(width))
print(f'=' * width)
