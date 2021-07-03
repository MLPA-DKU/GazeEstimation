from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import collections

import numpy as np
import torch


def load_batch(
        batch: Sequence[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False
    ):
    return [b.to(device=device, non_blocking=non_blocking) for b in [*batch]] if device is not None else batch


def update(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: Union[Callable, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        return_form: Callable = lambda i, t, o, l: (o, t)
    ):
    def __update(batch: Sequence[torch.Tensor]):
        model.train()
        optimizer.zero_grad()
        inputs, targets = load_batch(batch, device=device, non_blocking=non_blocking)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        return return_form(inputs, targets, outputs, loss)
    return __update


def evaluate(
        model: torch.nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        return_form: Callable = lambda i, t, o: (o, t)
    ):
    def __evaluate(batch: Sequence[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            inputs, targets = load_batch(batch, device=device, non_blocking=non_blocking)
            outputs = model(inputs)
        return return_form(inputs, targets, outputs)
    return __evaluate


class Engine:

    def __init__(
            self,
            process_fn: Callable,
            evaluators: Optional[Union[Callable, List[Callable]]] = None,
        ):
        self.process_fn = process_fn
        self.evaluators = evaluators

    def __call__(
            self,
            dataloader: Iterable,
        ):
        stacked_results = collections.defaultdict(list)
        for idx, batch in enumerate(dataloader):
            batch_outputs = self.process_fn(batch)
            batch_results = [evaluator(batch_outputs) for evaluator in self.evaluators]
            for i, r in enumerate(batch_results):
                stacked_results[f'{i}'].append(r.item() if isinstance(r, torch.Tensor) else r)
        return stacked_results


class StreamHandler:

    def __init__(
            self,
            total_epoch,
            total_steps,
            mode: str = 'period',
            tags: Optional[Union[str, List[str]]] = None,
            format_spec: str = '.3f'
        ):
        self.total_epoch = total_epoch
        self.total_steps = total_steps
        self.mode = {'period': lambda x: np.nanmean(x), 'latest': lambda x: x[-1]}[mode]
        self.tags = tags
        self.format_spec = format_spec
        self.header = self.__header__()

    def __call__(
            self,
            epoch_dix: int,
            batch_idx: int,
            values: Union[List, Dict[List]],
        ):
        head = self.header(epoch_dix, batch_idx)
        if self.tags is not None:
            assert len(self.tags) == len(values), 'mismatch of length between tags and values is founded.'
            tail = ' - '.join([f'{t}: {self.mode(v):{self.format_spec}}' for t, v in zip(self.tags, values)])
        else:
            tail = ' - '.join([f'{v:{self.format_spec}}' for v in values])
        full = ' - '.join([head, tail])
        print(f'\r{full}', end='')

    def __header__(self):
        total_epoch = self.total_epoch; digit_epoch = len(str(self.total_epoch))
        total_steps = self.total_steps; digit_steps = len(str(self.total_steps))
        def __header(epoch, batch):
            return f'Epoch[{epoch+1:>{digit_epoch}d}/{total_epoch}] - batch[{batch+1:>{digit_steps}d}/{total_steps}]'
        return __header


if __name__ == '__main__':

    def fun(x):
        return x ** 2

    def foo(x):
        return x + 1

    def bar(x):
        return x + 2

    fakeloader = [i for i in range(10)]

    engineer = Engine(fun, [foo, bar])
    building = engineer(fakeloader)
    breakpoint()
