from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch


def load_batch(
        batch: Sequence[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False
    ):
    return [b.to(device=device, non_blocking=non_blocking) for b in [*batch]] if device is not None else batch


def update_function(
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


def evaluation_function(
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


def epoch_function(
        fn: Callable,
        dataloader: Iterable,
        params,
    ):
    params.dataloader = dataloader
    for idx, batch in enumerate(dataloader):
        r = fn(batch)
        r = {k: v(r) for k, v in params.metrics.items()}


class Engine:

    def __init__(
            self,
            train_fn: Callable,
            valid_fn: Callable,
            engine_config: Dict[str, Callable]
        ):
        self.train_fn = train_fn
        self.valid_fn = valid_fn
        self.engine_params = EngineParams(engine_config)

    def __call__(
            self,
            trainloader: Iterable,
            validloader: Iterable,
        ):
        epoch_function(self.train_fn, trainloader, self.engine_params)
        epoch_function(self.valid_fn, validloader, self.engine_params)


class EngineParams:

    def __init__(self, config):
        self.state = 'init'
        self.epoch = 0
        self.batch = 0
        self.epochs = config.epochs
        self.metrics = config.metrics
        self.dataloader = None
