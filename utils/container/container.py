class Container(dict):

    def __init__(self, *args, **kwargs):
        super(Container, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if type(v) is dict:
                self[k] = Container(v)

    def __getattr__(self, key):
        def __proxy__(_dict, key):
            for k, v in _dict.items():
                if k == key:
                    return v
                if isinstance(v, Container):
                    ret = __proxy__(v, key)
                    if ret is not None:
                        return ret
                    else:
                        continue
        try:
            return __proxy__(self, key)
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, key, value):
        def __proxy__(_dict, key, value):
            for k in _dict.keys():
                if k == key:
                    _dict[k] = value
                    return True
                if isinstance(_dict[k], Container):
                    return __proxy__(_dict[k], key, value)
        if __proxy__(self, key, value):
            return
        self[key] = value

    def __delattr__(self, key):
        def __proxy__(_dict, key):
            for k in _dict.keys():
                if k == key:
                    del _dict[k]
                    return
                if isinstance(_dict[k], Container):
                    __proxy__(_dict[k], key)
        try:
            __proxy__(self, key)
        except KeyError as e:
            raise AttributeError(e)
