import gc
import torch


def denorm(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def load_batch(batch, device=None, non_blocking=False):
    batch = [*batch]
    batch = [b.to(device=device, non_blocking=non_blocking) for b in batch] if device is not None else batch
    return batch


def salvage_memory():
    torch.cuda.empty_cache()
    gc.collect()
