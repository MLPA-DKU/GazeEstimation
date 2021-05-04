from typing import Union
import os
import subprocess
import warnings

import torch.cuda


class NVIDIAGPUMEMInfo:

    def __init__(self, verbose=0, encoding='byte'):
        self.verbose = verbose
        self.__env__()
        self.nvidia_gpu_memory_total = self.__smi__(query='--query-gpu=memory.total', encoding=encoding)
        self.nvidia_gpu_memory_usage = self.__smi__(query='--query-gpu=memory.used', encoding=encoding)

    def __env__(self):
        if self.verbose >= 1:
            if 'CUDA_DEVICE_ORDER' not in os.environ or 'PCI_BUS_ID' != os.environ['CUDA_DEVICE_ORDER']:
                warn = 'It`s recommended to set ``CUDA_DEVICE_ORDER`` to be ``PCI_BUS_ID`` ' \
                       'by ``export CUDA_DEVICE_ORDER=PCI_BUS_ID``; ' \
                       'Otherwise, it`s not guaranteed that the GPU index from PyTorch ' \
                       'to be consistent the ``nvidia-smi`` results.'
                warnings.warn(warn)

    def __smi__(self, query, encoding):
        res = subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader'])
        res = res.decode().strip().split('\n')
        res = {x: self.convert_to_bytes(y) if encoding == 'byte' else y for x, y in enumerate(res)}
        return res

    @staticmethod
    def convert_to_bytes(memory_size):
        size, suffix = memory_size.split(' ')
        size = int(size)
        lookup = {
            'KB': size << 10, 'KiB': size << 10,
            'MB': size << 20, 'MiB': size << 20,
            'GB': size << 30, 'GiB': size << 30,
        }
        return lookup[suffix]


class DeviceAutoAllocator:

    # TODO: add param `required_minimum`

    def __init__(self):
        self.device_dict = NVIDIAGPUMEMInfo().nvidia_gpu_memory_usage
        self.device_dict_sorted = sorted(self.device_dict.items(), key=lambda item: item[1])

    def __call__(self, device_required=1):
        assert device_required <= len(self.device_dict_sorted), 'You are trying to allocate GPUs more than you have.'
        if torch.cuda.is_available():
            device = self.device_dict_sorted[0:device_required]
            device = f'cuda:{device[0][0]}' if device_required == 1 else [d[0] for d in device]
            return device
        else:
            return 'cpu'


def auto_device(parallel: Union[bool, int] = False):
    """Recommend GPU device by GPU memory usage.

    Args:
        parallel: setting number of GPUs to use when using torch.nn.DataParallel.
            If False, only 1 GPU with the least GPU memory usage is selected.

    Examples:
        >>> # Allocate to 1 GPU
        >>> device = auto_device(parallel=False)
        >>> model = torch.nn.Module()
        >>> model.to(device)
        >>> # Allocate to 4 GPU with torch.nn.DataParallel
        >>> device = auto_device(parallel=True)
        >>> model = torch.nn.Module()
        >>> model = torch.nn.DataParallel(model, device_ids=auto_device(parallel=4))
        >>> model.to(device)
        >>> # Allocate to every GPU with torch.nn.DataParallel
        >>> device = auto_device(parallel=True)
        >>> model = torch.nn.Module()
        >>> model = torch.nn.DataParallel(model)
        >>> model.to(device)
    """
    assert isinstance(parallel, bool) or isinstance(parallel, int)
    __alloc = DeviceAutoAllocator()
    if isinstance(parallel, bool):
        return __alloc(device_required=1)
    elif isinstance(parallel, int):
        return __alloc(device_required=parallel)
