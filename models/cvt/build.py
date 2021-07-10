from models.cvt.registry import model_entrypoints
from models.cvt.registry import is_model


def build_model(config, **kwargs):
    model_name = config.MODEL.NAME
    if not is_model(model_name):
        raise ValueError(f'Unknown model: {model_name}')
    return model_entrypoints(model_name)(config, **kwargs)
