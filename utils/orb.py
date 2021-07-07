import logging

import torch.utils.tensorboard


def switch_mode():
    if logging.StreamHandler.terminator == '\n':
        logging.StreamHandler.terminator = ''
    elif logging.StreamHandler.terminator == '':
        logging.StreamHandler.terminator = '\n'
    else:
        raise Exception


class Orb:

    def __init__(self, volume):
        logging.debug('initializing orb to visualize model results...')
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=volume)

    def log(self, tag, value, global_step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)
