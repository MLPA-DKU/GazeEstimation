import logging


def setup_logger(level=logging.INFO):
    head = '\r[%(asctime)-15s] (%(filename)s:line %(lineno)d) %(name)s:%(levelname)s :: %(message)s'
    logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(level)


def initializer(config, name, module, *args, **kwargs):
    module_name = config[name]['name']
    module_args = dict(config[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'config file takes precedence. overwriting is not allowed.'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def bootstrapping(config):
    pass
