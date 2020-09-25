import torch


class InferenceBenchmark:

    def __init__(self, iterations=10, warm_up=5, device=None, print_width=40):
        self.iterations = iterations
        self.warm_up = warm_up
        self.device = device
        self.print_width = print_width
        self.cpu_time_ms = 0
        self.gpu_time_ms = 0
        self.total_infer = 0

    def __call__(self, model, batch_size=8, input_shape=None, arch=None, device_name=None):
        model.to(self.device)
        model.eval()

        tensor_shape = [batch_size]
        tensor_shape.extend(list(input_shape))
        tensor = torch.rand(tensor_shape).to(self.device)

        for _ in range(self.warm_up):
            with torch.no_grad():
                model(tensor)

        for i in range(self.iterations):
            with torch.no_grad():
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    model(tensor)
                self.cpu_time_ms += prof.self_cpu_time_total / 1000
                self.gpu_time_ms += sum([event.cuda_time_total for event in prof.key_averages()]) / 1000

        self.cpu_time_ms /= self.iterations * batch_size
        self.gpu_time_ms /= self.iterations * batch_size
        self.total_infer = self.cpu_time_ms + self.gpu_time_ms

        self.display(arch, device_name, input_shape)

    def display(self, arch, device_name, input_shape):
        print(f'=' * self.print_width)
        print(f'{arch}'.center(self.print_width)) if arch is not None else None
        print(f'{device_name}'.center(self.print_width)) if device_name is not None else None
        print(f'image resolution: {input_shape}'.center(self.print_width))
        print(f'inference time (cpu): {self.cpu_time_ms:7.1f} ms'.center(self.print_width))
        print(f'inference time (gpu): {self.gpu_time_ms:7.1f} ms'.center(self.print_width))
        print(f'inference time (total): {self.total_infer:5.1f} ms'.center(self.print_width))
        print(f'frames per second (fps): {1000.0 // self.total_infer:7.1f}'.center(self.print_width))
        print(f'=' * self.print_width)


if __name__ == '__main__':

    import torchvision.models as models

    model = models.resnet18()
    shape = (3, 224, 224)

    benchmark = InferenceBenchmark(device='cpu')
    benchmark(model, input_shape=shape, arch='ResNet18', device_name='Intel i9-9980XE')
