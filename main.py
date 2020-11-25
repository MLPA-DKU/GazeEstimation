
epochs = 1000


def main():

    trainset = ...
    validset = ...
    test_set = ...
    trainloader = ...
    validloader = ...
    test_loader = ...

    model = ...

    criterion = ...
    evaluator = ...
    optimizer = ...
    scheduler = ...

    callbacks = ...

    for epoch in range(epochs):
        train()
        validate()
        if callbacks:
            break
        test()


def train():
    pass


def validate():
    pass


def test():
    pass


if __name__ == '__main__':
    main()
