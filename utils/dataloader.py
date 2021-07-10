import logging

import torch
import torch.utils.data


def auto_batch_size(
        dataset: torch.utils.data.Dataset,
        updater,
    ) -> int:

    def run(batch_size):
        logging.debug(f'trying to run with given batch size ({batch_size})')
        torch.cuda.empty_cache()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        try:
            for idx, batch in enumerate(dataloader):
                updater(batch)
                if idx == 2:
                    return True
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logging.debug('not enough GPU memory')
                return False
            else:
                raise e

    candidate = None
    for n in range(10**10):
        candidate = 2 ** n
        is_runnable = run(candidate)
        if not is_runnable:
            break
    candidate = candidate // 2
    logging.debug(f'the recommended batch_size is {candidate}')

    return candidate
