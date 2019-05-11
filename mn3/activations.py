from torch import nn


class HSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = self.relu(x + 3.) / 6.
        return out


class HSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.hsigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.hsigmoid(x)
