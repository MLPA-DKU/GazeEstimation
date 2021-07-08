from typing import List, Union
import os
import logging
import subprocess
import torch.cuda


def auto_device(
        num_device_desired: int = 1,
    ) -> Union[int, str, torch.device, List[int]]:
    logging.debug(f'trying to search for {num_device_desired} allocatable GPU(s)...')
    __alloc = DeviceAutoAllocator()
    try:
        device = __alloc(device_required=num_device_desired)
        logging.debug(f'finding {num_device_desired} device for allocating successfully')
        return device
    except Exception as e:
        logging.error(f'error occurs when searching device for allocate by "{e}"')


class NvidiaGPUMemoryInfo:

    def __init__(
            self,
            encoding: str = 'byte',
            verbose: int = 0,
        ):
        self.nvidia_gpu_memory_total = self.__smi__(query='--query-gpu=memory.total', encoding=encoding)
        self.nvidia_gpu_memory_usage = self.__smi__(query='--query-gpu=memory.used', encoding=encoding)
        self.verbose = verbose
        self.__env__()

    def __env__(self):
        if self.verbose >= 1:
            if 'CUDA_DEVICE_ORDER' not in os.environ or 'PCI_BUS_ID' != os.environ['CUDA_DEVICE_ORDER']:
                warn = 'It`s recommended to set ``CUDA_DEVICE_ORDER`` to be ``PCI_BUS_ID`` ' \
                       'by ``export CUDA_DEVICE_ORDER=PCI_BUS_ID``; ' \
                       'Otherwise, it`s not guaranteed that the GPU index from PyTorch ' \
                       'to be consistent the ``nvidia-smi`` results.'
                logging.warning(warn)

    def __smi__(
            self,
            query: str,
            encoding: str = 'byte'
        ):
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

    def __init__(self):
        self.device_info = NvidiaGPUMemoryInfo()
        self.device_used = self.device_info.nvidia_gpu_memory_usage
        self.device_dict_sorted = sorted(self.device_used.items(), key=lambda item: item[1])

    def __call__(
            self,
            device_required: int = 1
        ):
        assert device_required <= len(self.device_dict_sorted), 'you are trying to allocate GPUs more than you have'
        if torch.cuda.is_available():
            if device_required >= 1:
                device = self.device_dict_sorted[0:device_required]
                device = f'cuda:{device[0][0]}' if device_required == 1 else [d[0] for d in device]
                logging.debug(f'finding {device_required} GPU(s) for allocating successfully')
                return device
            elif device_required == 0:
                logging.info(f'it looks like you do not need a GPU, therefore we switch your device to cpu')
                return 'cpu'
            else:
                raise ValueError
        else:
            logging.warning('GPU not detected\nswitch your device to cpu automatically')
            return 'cpu'
