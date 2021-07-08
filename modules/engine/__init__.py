from typing import Callable, Optional, Sequence, Tuple, Union
from modules.engine.engine import Engine

import torch
import torch.utils.data


def load_batch(
        batch: Sequence[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False
    ) -> Tuple[Union[torch.Tensor, Sequence, str, bytes], ...]:
    return (b.to(device=device, non_blocking=non_blocking) for b in [*batch]) if device is not None else batch


def update(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: Union[Callable, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        return_form: Callable = lambda i, t, o, l: (o, t)
    ) -> Callable:
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
    ) -> Callable:
    def __evaluate(batch: Sequence[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            inputs, targets = load_batch(batch, device=device, non_blocking=non_blocking)
            outputs = model(inputs)
        return return_form(inputs, targets, outputs)
    return __evaluate
