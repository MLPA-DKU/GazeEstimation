import logging

import torch.utils.tensorboard


class Orb:

    def __init__(self, volume):
        logging.debug('initializing orb to visualize model results...')
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=volume)

    def log(self, tag, value, global_step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)
