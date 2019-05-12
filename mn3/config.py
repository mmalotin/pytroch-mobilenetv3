from dataclasses import dataclass
from typing import List
from copy import deepcopy


@dataclass
class BackboneConfig:
    out_channels: List[int]
    expansion_sizes: List[int]
    strides: List[int]
    kernels: List[int]
    nonlinearities: List[str]
    ses: List[bool]

    def scale_width(self, scale=1, inplace=False):
        tmp = self if inplace else deepcopy(self)
        tmp.out_channels[1:] = [int(x * scale)
                                for x in tmp.out_channels[1:]]
        tmp.expansion_sizes[1:] = [int(x * scale)
                                   for x in tmp.expansion_sizes[1:]]

        return tmp


@dataclass
class MobilenetConfig:
    bbone_conf: BackboneConfig
    head_channels: List[int]
    add_bn: bool

    def scale_width(self, scale=1, inplace=False):
        tmp = self if inplace else deepcopy(self)
        tmp.bbone_conf.scale_width(scale, inplace=True)
        tmp.head_channels = [int(x * scale) for x in tmp.head_channels]

        return tmp


LARGE_BBONE = BackboneConfig(
    kernels=[3]*3 + [5]*3 + [3]*6 + [5]*3,
    out_channels=[16, 24, 24, 40, 40, 40, 80, 80,
                  80, 80, 112, 112, 160, 160, 160],
    expansion_sizes=[16, 64, 72, 72, 120, 120, 240, 200,
                     184, 184, 480, 672, 672, 672, 960],
    ses=[False]*3 + [True]*3 + [False]*4 + [True]*5,
    nonlinearities=['relu6']*6 + ['hswish']*9,
    strides=[1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2]
    )

SMALL_BBONE = BackboneConfig(
    kernels=[3]*3 + [5]*8,
    out_channels=[16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
    expansion_sizes=[16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576],
    ses=[True, False, False] + [True]*8,
    nonlinearities=['relu6']*3 + ['hswish']*8,
    strides=[2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1]
    )

LARGE = MobilenetConfig(
    bbone_conf=LARGE_BBONE,
    head_channels=[960, 1280],
    add_bn=False
    )

SMALL = MobilenetConfig(
    bbone_conf=SMALL_BBONE,
    head_channels=[576, 1280],
    add_bn=True
    )
