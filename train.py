import modules


def main(config):
    model, optimizer, criterion, evaluator, interface, device = modules.bootstrapping(config)
    train_function = modules.update(model, optimizer, criterion, device)
    valid_function = modules.evaluate(model, device)
    trainloader, validloader = modules.bootstrapping_dataloader(config, train_function)
