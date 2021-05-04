from typing import Union
import os
import uuid
import socket
import numpy as np
import torch.utils.tensorboard


class R6DB:

    base_folder = 'R6'

    def __init__(self, f: Union[str, os.PathLike], uniq: Union[str, int, uuid.UUID]):
        self.f = os.path.join(f, self.base_folder, f'{self.base_folder}.{uniq}' if uniq else f'{self.base_folder}')
        self.f = os.path.abspath(self.f)
        if not os.path.exists(self.f):
            os.makedirs(self.f)
        self.uniq = uniq
        self.machine = socket.gethostname()


# TODO: PyTorch Integrated Checkpoint Module - Inspired by CheckFreq from Microsoft Project Fiddle
class R6Checkpoint(R6DB):

    def __init__(self, obj, f, uniq=None):
        self.base_folder = 'checkpoint'
        super(R6Checkpoint, self).__init__(f=f, uniq=uniq)

        self.obj = obj

    def __tape__(self, suffix):
        return f'checkpoint.R6.{self.uniq}.{suffix}.pth'

    def __save__(self, suffix):
        torch.save(self.obj, os.path.join(self.f, self.__tape__(suffix)))

    def __load__(self, suffix):
        return torch.load(os.path.join(self.f, self.__tape__(suffix)))

    def save(self, suffix):
        self.__save__(suffix)

    def load(self, suffix):
        return self.__load__(suffix)


class R6Tensorboard(R6DB):

    def __init__(self, f, uniq=None):
        self.base_folder = 'tensorboard'
        super(R6Tensorboard, self).__init__(f=f, uniq=uniq)

        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.f)


class R6EarlyStopping:

    def __init__(self, monitor=None, patience=0, delta=None):
        self.monitor = monitor
        self.patience = patience
        self.delta = delta if delta is not None else 0

        self.best_score = np.inf
        self.counter = 1

    def __call__(self):
        if self.counter == self.patience:
            quit()
        self.counter = 1 if self.monitor < self.best_score - self.delta else self.counter + 1
        self.best_score = min(self.monitor, self.best_score - self.delta)


class R6MetricHandler:

    def __init__(self, metric, format_spec='.f'):
        self.metric = metric
        self.format_spec = format_spec
        self.__dict__[self.metric] = []

    def __call__(self, value):
        self.__dict__[self.metric].append(value)

    def __str__(self):
        return f'{self.metric}: {format(np.nanmean(self.__dict__[self.metric]), self.format_spec)}'


class R6IX:

    def __init__(self, epochs, len_trainloader, len_validloader, device=None):

        # universal
        self.epochs = epochs
        self.train_batches = len_trainloader
        self.valid_batches = len_validloader
        self.batches = self.train_batches
        self.device = device if device is not None else 'cpu'
        self.mode = 'train'

        # variable
        self.epoch = 0
        self.batch = 0

        # format
        self.format_spec_epoch = f'>{len(str(self.epochs))}d'
        self.format_spec_batch = f'>{max(len(str(self.train_batches)), len(str(self.valid_batches)))}d'
        self.format_spec_metrics = f'.3f'

        self.epoch_str = format(self.epoch, self.format_spec_epoch)
        self.batch_str = format(self.batch, self.format_spec_batch)
        self.batches_str = format(self.batches, self.format_spec_batch)

    def train(self):
        self.mode = 'train'
        self.batches = self.train_batches
        self.batches_str = format(self.batches, self.format_spec_batch)

    def eval(self):
        self.mode = 'valid'
        self.batches = self.valid_batches
        self.batches_str = format(self.batches, self.format_spec_batch)


class R6BatchManager:

    def __init__(self, r6ix: R6IX):
        self.r6ix = r6ix
        self.metric_handlers = {}

    def add_metric(self, metric):
        self.metric_handlers[metric] = R6MetricHandler(metric, self.r6ix.format_spec_metrics)

    def update(self, metric, value):
        self.metric_handlers[metric](value)


class R6Printer:

    def __init__(self, r6ix: R6IX, r6_batch: R6BatchManager):
        self.r6ix = r6ix
        self.r6_batch = r6_batch

    def __format_epoch__(self):
        return f'Epoch[{self.r6ix.epoch_str}/{self.r6ix.epochs}]'

    def __format_batch__(self):
        return f'batch[{self.r6ix.batch_str}/{self.r6ix.batches_str}]'

    def __format_handler__(self):
        return [str(v) for _, v in self.r6_batch.metric_handlers.items()]

    def __message__(self):
        m = [self.__format_epoch__(), self.__format_batch__()]
        m.extend(self.__format_handler__())
        return ' - '.join(m)

    def display(self):
        print(f'\r{self.__message__()}', end='')


def print_result_on_epoch_end(epoch, epochs, scores):
    print(f'\n[ RES ] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
          f'angular error [{np.nanmean(scores):.3f}|{np.nanstd(scores):.3f}|'
          f'{np.min(scores):.3f}|{np.max(scores):.3f}:MEAN|STD|MIN|MAX]')
