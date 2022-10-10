from .resnet import FlatResNet224, Policy224
from .resnet import FlatResNet32, Policy32

from .base import BasicBlock


def cifar10_blockdrop_110():
    layer_config = [18, 18, 18]
    rNet = resnet.FlatResNet32(BasicBlock, layer_config, num_classes=10)
    agent = resnet.Policy32([1, 1, 1], num_blocks=54)
    return rNet, agent


def imagenet_blockdrop_101():
    layer_config = [3, 4, 23, 3]
    rNet = resnet.FlatResNet224(base.Bottleneck, layer_config, num_classes=1000)
    agent = resnet.Policy224([1, 1, 1, 1], num_blocks=33)
    return rNet, agent


def cifar100_blockdrop_110():
    layer_config = [18, 18, 18]
    rNet = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=100)
    agent = resnet.Policy32([1, 1, 1], num_blocks=54)
    return rNet, agent
