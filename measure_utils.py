import torch.nn as nn
import time
from tqdm import tqdm

from utils import *


class FConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FConv2d, self).forward(x)
        output_area = output.size(-1) * output.size(-2)
        filter_area = np.prod(self.kernel_size)
        self.num_ops += 2 * self.in_channels * self.out_channels * filter_area * output_area
        return output


class FLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FLinear, self).__init__(in_features, out_features, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FLinear, self).forward(x)
        self.num_ops += 2 * self.in_features * self.out_features
        return output


nn.Conv2d = FConv2d
nn.Linear = FLinear


def get_overhead(power, model, device, iter_num, img):
    s_t, s_flops = 0.0, 0
    power.start()
    for _ in range(iter_num):
        t1 = time.time()
        _, flop = model.adaptive_forward(img, device)
        t2 = time.time()
        s_t += (t2 - t1)
        s_flops += flop
    is_s, energy_cost, tm = power.getTotalEnergy()
    power.reset()
    if is_s:
        return True, s_flops/1000000, s_t, energy_cost
    else:
        return False, None, None, None


def test_flops(dataloader, model, device):
    results = []
    for img in dataloader:
        _, flop = model.adaptive_forward(img, device)
        results.append(flop)
    return results


def test_TX2_metric(dataloader, model, device, power, max_test_num, repeat_num):
    result_list = []
    for i, x in tqdm(enumerate(dataloader)):
        if i >= max_test_num:
            break
        if type(x) == list:
            x = x[0]
        x = x.to(device)
        is_success, sum_flops, sum_t, energy = get_overhead(power, model, device, repeat_num, x)
        if is_success:
            result_list.append([sum_flops/repeat_num, sum_t/repeat_num, energy/repeat_num])
            assert sum_flops != 0
    return result_list


def get_repeat_num(model_name, task_name=None):
    if model_name == 'RANet':
        repeat_num = 50
    elif model_name == 'DeepShallow':
        repeat_num = 40
    elif model_name == 'SkipNet':
        repeat_num = 10
    else:
        repeat_num = 20
    return repeat_num