import os.path

import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import csv


from utils import *
from predict_tx2 import PowerLogger


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


def get_overhead(iter_num, img):
    s_t, s_flops = 0.0, 0
    power.start()
    for _ in range(iter_num):
        t1 = time.time()
        _, flop = model.adaptive_forward(img, device)
        t2 = time.time()
        s_t += (t2 - t1)
        s_flops += flop
    power.stop()
    is_s, energy_cost, tm = power.getTotalEnergy()
    power.reset()
    if is_s:
        return True, s_flops, s_t, energy_cost
    else:
        return False, None, None, None


repeat_num = 200
max_test_num = 5
task_name_list = [
    #('CIFAR100', 'DeepShallow'),
    #('TinyImageNet', 'DeepShallow'),
    ('CIFAR10', 'SkipNet'),
    ('CIFAR10', 'BlockDrop'),
    ('CIFAR100', 'RANet'),
]
per_dict = {
    'inf': [0.03, 0.06, 0.09, 0.12, 0.15],
    '2': [10, 15, 20, 25, 30]
}
print(repeat_num, max_test_num)

if not os.path.isdir('results'):
    os.mkdir('results')

final_results = {}
final_save_path = 'results/AbuseGanDeploy.res'
for device_id in ['cpu', 'cuda']:
    device = torch.device(device_id)
    for (data_name, model_name) in task_name_list:
        model, _, _ = load_model_data(model_name, data_name)
        model = model.to(device).eval()
        for norm in ['inf', '2']:
            for per_size in per_dict[norm]:

                power = PowerLogger(interval=0.1)

                sub_dir = model_name + '_' + data_name + '_' + norm + '_' + 'hyper0.001_' + str(per_size)
                adv_img = torch.load('AbuseGanPerturbation/' + sub_dir + '/perturbation.per')
                test_loader = DataLoader(adv_img, batch_size=1, shuffle=False)

                result_list = []
                for i, x in tqdm(enumerate(test_loader)):
                    if i > max_test_num:
                        break
                    x = x.to(device)
                    is_success, sum_flops, sum_t, energy = get_overhead(repeat_num, x)
                    if is_success:
                        result_list.append([sum_flops, sum_t, energy])
                final_results[sub_dir + '_' + device_id] = result_list
                torch.save(final_results, final_save_path)
                print(sub_dir + '_' + device_id, 'successful')

torch.save(final_results, final_save_path)
