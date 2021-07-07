import logging

import torch


def save_checkpoint(checkpoint, f):
    logging.info(f'trying to save checkpoint to {f}...')
    try:
        # torch.save(checkpoint, f)
        boot.this_fails()
        logging.info(f'saving checkpoint at {f} successfully')
    except Exception:
        logging.error('error occurs when saving checkpoint')


def load_checkpoint(f, device=None):
    logging.info(f'loading checkpoint from {f}...')
    try:
        checkpoint = torch.load(f, map_location=device if device is not None else 'cpu')
    except Exception:
        logging.error('error occurs when loading checkpoint')


class Crate:

    def __init__(self):
        pass


if __name__ == '__main__':
    import modules.engine.bootstrap as boot
    boot.setup_logger()
    obj = torch.rand((1, 3, 224, 224))
    save_checkpoint(obj, './temp.pth')
