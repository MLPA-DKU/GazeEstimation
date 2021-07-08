import logging
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.utils.data

import utils


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


class EngineParameters:

    def __init__(self):
        self.epoch = 0
        self.global_steps = 0
        self.is_train = False
        self.batch_idx = 0
        self.batch_total = None
        self.batch_outputs = None


class Engine:

    def __init__(
            self,
            process_fn: Callable,
            evaluators: Union[Callable, List[Callable]],
            writer: utils.Orb = None,
            train: bool = False,
        ):
        self.process_fn = process_fn
        self.evaluators = evaluators if isinstance(evaluators, list) else [evaluators]
        self.engine_params = EngineParameters()
        self.engine_params.is_train = train
        self.writer = writer

    def __call__(
            self,
            dataloader: torch.utils.data.DataLoader,
            tags: Union[str, List[str], Tuple[str]],
        ):
        self.__run_at_the_init_of_epoch__()
        self.engine_params.batch_total = len(dataloader)
        for idx, batch in enumerate(dataloader):
            self.__run_at_the_init_of_batch__()
            outputs = self.process_fn(batch)
            self.engine_params.batch_idx = idx + 1
            self.engine_params.batch_outputs = [evaluator(*outputs) for evaluator in self.evaluators]
            self.__run_at_the_term_of_batch__(tags)
        self.__run_at_the_term_of_epoch__()

    def __run_at_the_init_of_epoch__(self):
        self.engine_params.epoch += 1
        logging.info(f'starting epoch {self.engine_params.epoch:>5d}...')
        logging.StreamHandler.terminator = ''

    def __run_at_the_init_of_batch__(self):
        pass

    def __run_at_the_term_of_batch__(self, tags):
        header = f'training' if self.engine_params.is_train else f'validation'
        stream = f'{header} session is proceeding: ' \
                 f'{self.engine_params.batch_idx:>{len(str(self.engine_params.batch_total))}d}/' \
                 f'{self.engine_params.batch_total}'
        logging.info(stream)
        tags = [f'{header}_{tag}' for tag in tags]
        self.writer.log(tags, self.engine_params.batch_outputs, self.engine_params.global_steps)
        self.engine_params.global_steps += 1

    def __run_at_the_term_of_epoch__(self):
        logging.StreamHandler.terminator = '\n'
        logging.info(f'session for epoch {self.engine_params.epoch:>5d} is done')
