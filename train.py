import modules


def main(config):
    engine = modules.bootstrapping(config)
    engine.build()


if __name__ == '__main__':
    main(...)
