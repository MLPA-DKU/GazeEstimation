import torch.nn as nn


class FooLoss(nn.Module):

    def __init__(self):
        super(FooLoss, self).__init__()

    def forward(self, x):
        return x ** 2


if __name__ == '__main__':
    foo = FooLoss()
    print(foo(2))
