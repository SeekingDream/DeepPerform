from .PolicyNet import SkipNet
from .PolicyNet import PolicyNet
from .PolicyNet import RasPolicyNet
from .PolicyNet import DeepShallowPolicyNet


from .skipnet.cifar10 import cifar10_rnn_gate_rl_110
from .skipnet.cifar10 import cifar100_rnn_gate_rl_110
from .skipnet.imageNet import imagenet_rnn_gate_rl_101


from .blockdrop import cifar10_blockdrop_110
from .blockdrop import cifar100_blockdrop_110
from .blockdrop import imagenet_blockdrop_101


from .RANet import RANet
from .RANet.adaptive_inference import dynamic_evaluate
from .RANet.adaptive_inference import Tester


# from .DeepShallow import WideResNet_SDN
# from .DeepShallow import MobileNet_SDN
# from .DeepShallow import ResNet_SDN
# from .DeepShallow import VGG_SDN
from .DeepShallow import load_deepshallow_model


