import time
import random
import tqdm
import numpy as np

epochs = 1000


def train(dataloader, model, epoch, batch_size=50):

    batch_char = len(str(batch_size))
    r_bar = 'BATCH {n:' + f'{batch_char}d' + '}/{total:' + f'{batch_char}d' + '}\t ETA: {remaining}{postfix}'
    bar_format = '{l_bar}{bar:30}| ' + r_bar

    epochs_char = len(str(epochs))
    desc = f'TRAIN EPOCH {epoch + 1:{epochs_char}d}/{epochs:{epochs_char}d}'

    dataloader = tqdm.tqdm(iterable=dataloader,
                           desc=desc,
                           ncols=200,
                           bar_format=bar_format)

    ret = []
    for i, data in enumerate(dataloader):
        model(data)
        results = random.random()
        dataloader.set_postfix_str(f'loss: {results:.3f}, accuracy: {results:.3f}')
        time.sleep(0.1)
        ret.append(results)
    return ret


def func(x):
    return x


if __name__ == '__main__':

    for epoch in range(epochs):
        batch_size = 100
        t_loader = np.arange(batch_size).tolist()
        v_loader = np.arange(50).tolist()
        t_logs = train(t_loader, func, epoch, batch_size)
