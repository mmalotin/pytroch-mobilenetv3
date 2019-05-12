from .blocks import Bneck, ConvBNAct
from .activations import HSwish
from torch import nn

nls = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'hswish': HSwish
    }


class MobilenetBackbone(nn.Module):
    def __init__(self, conf):
        super().__init__()
        conv1 = ConvBNAct(3, 16, 3, bn=True, act=HSwish,
                          stride=2, padding=1, bias=False)

        blocks = self.__create_blocks(conf)

        self.fwd = nn.Sequential(conv1, *blocks)

    def forward(self, x):
        return self.fwd(x)

    def __create_blocks(self, conf):
        in_channels = [conf.out_channels[0]] + conf.out_channels[:-1]
        return [Bneck(inp=in_channels[i],
                      out=conf.out_channels[i],
                      ks=conf.kernels[i],
                      exp=conf.expansion_sizes[i],
                      stride=conf.strides[i],
                      nl=nls[conf.nonlinearities[i]],
                      se=conf.ses[i])
                for i in range(len(conf.out_channels))]


class MobilenetV3(nn.Module):
    def __init__(self, conf, n_classes):
        super().__init__()
        self.bbone = MobilenetBackbone(conf.bbone_conf)

        pool = nn.AdaptiveAvgPool2d(1)
        convs = self.__create_convs(conf, n_classes)

        self.head = nn.Sequential(convs[0], pool, *convs[1:])

    def forward(self, x):
        bs = x.size(0)
        out = self.bbone(x)
        out = self.head(out)

        return out.view(bs, -1)

    def __create_convs(self, conf, n_classes):
        channels = ([conf.bbone_conf.out_channels[-1]]
                    + conf.head_channels
                    + [n_classes])
        bns = [True] + [conf.add_bn] * 2

        convs = [ConvBNAct(channels[i], channels[i+1], 1,
                           bn=bns[i], act=HSwish, bias=False)
                 for i in range(len(channels) - 1)]

        return convs
