import os
import time
import subprocess

import numpy as np
import torch.cuda


class AverageMeter:

    def __init__(self, name=None, format_spec='.f'):
        self.name = name
        self.format_spec = format_spec
        self.values = []
        self.mean = None

    def __str__(self):
        return format(self.mean, self.format_spec)

    def step(self, value):
        self.values.append(value)
        self.mean = np.nanmean(self.values)

    def initialize(self):
        self.values = []
        self.mean = None


class CheckScore:

    def __init__(self, delta=0):
        self.delta = delta
        self.is_best = False
        self.best_score = np.inf

    def step(self, score):
        self.is_best = True if score < self.best_score - self.delta else False
        self.best_score = min(score, self.best_score - self.delta)


class CircuitBreaker:

    # early stopper

    def __init__(self, monitor=None, delta=0, patience=0, baseline=None, verbose=0):
        self.monitor = monitor
        self.patience = patience
        self.baseline = baseline
        self.verbose = verbose

        self.score_meter = CheckScore(delta=delta)
        self.counter = 1

    def __call__(self, monitor=None):

        if self.counter == self.patience:
            quit()

        value = self.monitor[-1] if self.monitor is not None else monitor
        self.score_meter.step(value)
        if self.baseline:
            self.score_meter.is_best = False if value < self.baseline else self.score_meter.is_best

        self.counter = 1 if self.score_meter.is_best else self.counter + 1

        if self.verbose > 0:
            print(f'Circuit breaker counter [{self.counter}/{self.patience}]')


def nvidia_memory_map():
    if 'CUDA_DEVICE_ORDER' not in os.environ or 'PCI_BUS_ID' != os.environ['CUDA_DEVICE_ORDER']:
        pass

    res = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'])
    gpu_mem = res.decode().strip().split('\n')
    gpu_mem_map = {x: y.split(',') for x, y in zip(range(len(gpu_mem)), gpu_mem)}
    return gpu_mem_map


def convert_to_bytes(size, suffix=''):
    size = float(size)
    if not suffix or suffix.isspace():
        return size
    size = int(size)
    suffix = suffix.lower()
    if suffix == 'kb' or suffix == 'kib':
        return size << 10
    elif suffix == 'mb' or suffix == 'mib':
        return size << 20
    elif suffix == 'gb' or suffix == 'gib':
        return size << 30
    return -1


def auto_device(metrics='memory', required_minimum=None, wait_time=20):
    assert (metrics == 'memory' or metrics == 'utils')
    if torch.cuda.is_available():
        gpu_mem_map = nvidia_memory_map()
        min_usage = float('inf')
        gpu_index = -1
        if required_minimum is not None:
            if 'memory' == metrics:
                req_min = required_minimum.split()
                req_min = convert_to_bytes(req_min[0], req_min[1])
            else:
                req_min = float(required_minimum.replace('%', ''))
        else:
            req_min = -1

        while min_usage < req_min or min_usage == float('inf'):
            for k, v in gpu_mem_map.items():
                if 'memory' == metrics:
                    v = v[0].split()
                    v = convert_to_bytes(v[0], v[1])
                else:
                    v = float(v[1].replace('%', ''))
                if v < min_usage:
                    min_usage = v
                    gpu_index = k
                if min_usage < req_min:
                    print(f'There is no gpu satisfying the required resource: {required_minimum} :(')
                    time.sleep(wait_time)
        return f'cuda:{gpu_index}'
    else:
        return -1


if __name__ == '__main__':
    device = auto_device()
    breakpoint()
