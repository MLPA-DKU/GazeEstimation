import importlib
import warnings


def is_valid_import(modules):
    succeeded = False
    for module in modules:
        try:
            importlib.import_module(module)
            succeeded = True
            break
        except ImportError:
            succeeded = False
    if not succeeded:
        message = "Warning: ... is configured to use, but currently not installed on this machine." \
                  "Please install ... with '...', upgrade ... to version >= 1.1 to use '...'" \
                  "or turn off the option in the 'config.json' file."
        warnings.warn(message)