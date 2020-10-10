from functools import wraps


class Tracker(object):

    def __init__(self):
        pass

    def on(self, event_name, *args, **kwargs):
        def decorator(func):
            self.add_event_handler(event_name, func, *args, **kwargs)
            return func
        return decorator

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        if isinstance(event_name, ...):
            for e in event_name:
                self.add_event_handler(e, handler, *args, **kwargs)
            return ...

        return ...
