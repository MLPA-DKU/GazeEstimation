import gc
import torch


def salvage_memory():
    torch.cuda.empty_cache()
    gc.collect()
