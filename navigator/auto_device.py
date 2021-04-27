import subprocess

import numpy as np
import torch.cuda


def nvidia_smi():
    # if 'CUDA_DEVICE_ORDER' not in os.environ or 'PCI_BUS_ID' != os.environ['CUDA_DEVICE_ORDER']:
    #     warnings.warn('It`s recommended to set ``CUDA_DEVICE_ORDER`` to be ``PCI_BUS_ID`` '
    #                   'by ``export CUDA_DEVICE_ORDER=PCI_BUS_ID``; '
    #                   'Otherwise, it`s not guaranteed that the GPU index from PyTorch '
    #                   'to be consistent the ``nvidia-smi`` results.')
    res = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'])
    res = res.decode().strip().split('\n')
    res = {f'cuda:{x}': y for x, y in enumerate(res)}
    return res


def convert_to_bytes(memory_size):
    size, suffix = memory_size.split(' ')
    size = int(size)
    lookup = {
        'KB': size << 10, 'KiB': size << 10,
        'MB': size << 20, 'MiB': size << 20,
        'GB': size << 30, 'GiB': size << 30,
    }
    return lookup[suffix]


def auto_device():
    if torch.cuda.is_available():
        device = None
        device_info = nvidia_smi()
        device_least_used = np.inf
        for device_id, device_usage in device_info.items():
            device_usage = convert_to_bytes(device_usage)
            if device_usage < device_least_used:
                device = device_id
                device_least_used = min(device_usage, device_least_used)
        return device
    else:
        return 'cpu'
