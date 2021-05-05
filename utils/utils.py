import random
import numpy as np
import torch
import torch.backends.cudnn


def enable_easy_debug(enable=False):
    torch.autograd.set_detect_anomaly(enable)


def enable_reproducibility(enable=False, random_seed=42):
    if enable:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
