import yaml


def parse_config(f):
    with open(f) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def serialize_config(obj, f):
    with open(f, 'w') as f:
        yaml.dump(obj, f)


def print_config(obj):
    print(yaml.dump(obj, allow_unicode=True, default_flow_style=False, sort_keys=False))


def initializer(config, name, module, *args, **kwargs):
    module_name = config[name]['name']
    module_args = dict(config[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'config file takes precedence. overwriting is not allowed.'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def bootstrap():
    pass


class Kernel:

    def __init__(self):
        pass


class Trainer():

    def __init__(self):
        pass
