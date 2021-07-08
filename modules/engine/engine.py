from typing import Callable, List, Union

import torch.utils.data


class Engine:

    def __init__(
            self,
            process_fn: Callable,
            evaluators: Union[Callable, List[Callable]],
        ):
        self.process_fn = process_fn
        self.evaluators = evaluators if isinstance(evaluators, list) else [evaluators]

    def __call__(
            self,
            dataloader: torch.utils.data.DataLoader,
        ):
        for idx, batch in enumerate(dataloader):
            outputs = self.process_fn(batch)
            outputs = list(evaluator(*outputs) for evaluator in self.evaluators)
