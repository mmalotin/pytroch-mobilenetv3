# MobilenetV3 in PyTorch

## Paper
[Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)

## Structure

- __mn3/activations.py__ - contains implementations of hard sigmoid and hard swish

- __mn3/blocks.py__ - contains implementations of squeeze-and-excitation block and of MobileNetV3 bottleneck block

- __mn3/config.py__ - contains cofiguration classes for MobileNetV3 backbone and MobileNetV3, you can create your own configuration or use onw of 4 predefined configurations (SMALL, LARGE, SMALL_BBONE, LARGE_BBONE)

- __mn3/nets.py__ - contains implementations of MobileNetV3 backbone and mobilnetv3 for classification (if you need just a backbone you can use backbone without classification head).

### Example 1 (how to create MobileNetV3 large for classification):

```python
from mn3.nets import MobilenetV3
import mn3.config as config

net = MobilenetV3(config.LARGE, n_classes=1000) # MobileNetV3 large
```

### Example 2 (how to create MobileNetV3 small backbone):
```python
from mn3.nets import MobilenetBackbone
import mn3.config as config

net = MobilenetBackbone(config.SMALL_BBONE) # MobileNetV3 small backbone
```

### Example 3 (how to scale width for network/backbone network):

```python
from mn3.nets import MobilenetV3
import mn3.config as config

small075 = config.SMALL.scale_width(0.75, inplace=False)

net = MobilenetV3(small075, n_classes=1000) # MobileNetV3 small with 0.75 width
```
