import numpy as np


def print_result(epoch, epochs, idx, dataloader, losses, scores, header=None):
    print(f'\r[{header}] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
          f'batch[{idx + 1:>{len(str(len(dataloader)))}}/{len(dataloader)}] - '
          f'loss: {np.nanmean(losses):.3f} - angular error: {np.nanmean(scores):.3f}', end='')


def print_result_on_epoch_end(epoch, epochs, scores):
    print(f'\n[ RES ] Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}] - '
          f'angular error (°) [{np.nanmean(scores):.3f}|{np.nanstd(scores):.3f}|'
          f'{np.min(scores):.3f}|{np.max(scores):.3f}:MEAN|STD|MIN|MAX]')


def format_epoch(epoch, epochs):
    return f'Epoch[{epoch + 1:>{len(str(epochs))}}/{epochs}]'


def format_batch(idx, dataloader):
    return f'batch[{idx + 1:>{len(str(len(dataloader)))}}/{len(dataloader)}]'


def format_value(tag, value, format_spec='.3f'):
    return f'{tag}: {value:{format_spec}}'



