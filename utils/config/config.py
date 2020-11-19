import os.path
import json

import utils


class ConfigParser(utils.Container):

    def __init__(self, filename):
        self.filename = os.path.abspath(filename)
        super(ConfigParser, self).__init__(self.read_configs())

    def read_configs(self):
        with open(self.filename) as f:
            configs = json.load(f)
        return configs

    def print_configs(self, _dict, indent=''):
        for k, v in _dict.items():
            if isinstance(_dict[k], utils.Container):
                print(indent+f'{k}:')
                self.print_json(v, indent+'    ')
            else:
                print(indent+f'{k}: {v}')

    def initialize_object(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_name for k in kwargs])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)



def initialize_object(config, module, package, *args, **kwargs):
    module_name = config[module]['type']
    module_args = config[module]['args']
    assert all([k not in module_name for k in kwargs])
    module_args.update(kwargs)
    return getattr(package, module_name)(*args, **module_args)
