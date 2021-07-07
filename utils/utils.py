import logging
import os.path
import random
import numpy as np
import torch
import torch.backends.cudnn
import tensorwatch as tw


def enable_easy_debug():
    torch.autograd.set_detect_anomaly(True)


def enable_reproducibility(manual_seed=42):
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(manual_seed)
    random.seed(manual_seed)


def count_trainable_params(model, unit='M', format_spec='.1f'):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    readable = {
        'M': lambda x: f'{x / 1000000:{format_spec}}M',
        'K': lambda x: f'{x / 1000:{format_spec}}K',
        'N': lambda x: f'{x}'
    }
    return readable[unit.upper()](params)


def summarize_model(model):
    try:
        num_params = count_trainable_params(model)
        logging.info(f'trainable parameters of given model: {num_params}')
    except Exception:
        logging.error('error occurs when counting trainable parameters of given model')

    try:
        logging.info('print model status powered by tensorwatch')
        summary = tw.model_stats(model, (1, 3, 224, 224))
        summary.to_html(os.path.join('./', 'summary.html'))
        logging.info(f'\n{summary.iloc[-1]}')
    except Exception:
        logging.error('error occurs when summarizing model status by tensorwatch')


if __name__ == '__main__':
    import torchvision.models as models
    import modules.engine.bootstrap as boot

    boot.setup_logger()
    model = models.resnet18()
    summarize_model(model)
