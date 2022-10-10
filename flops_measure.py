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


def get_overhead(iter_num, x, model, device):
    sum_t, sum_flops = 0.0, 0
    power.start()
    for _ in range(iter_num):
        t1 = time.time()
        _, flop = model.adaptive_forward(x, device)
        t2 = time.time()
        sum_t += (t2 - t1)
        sum_flops += flop
    power.stop()
    is_success, energy, tm = power.getTotalEnergy()
    power.reset()
    if is_success:
        return True, sum_flops, sum_t, energy
    else:
        return False, None, None, None


repeat_num = 10
max_test_num = 1000
task_name_list = [
    ('CIFAR100', 'DeepShallow'),
    ('TinyImageNet', 'DeepShallow'),
    ('CIFAR10', 'SkipNet'),
    ('CIFAR10', 'BlockDrop'),
    ('CIFAR100', 'RANet'),
]
print(repeat_num, max_test_num)

if not os.path.isdir('results'):
    os.mkdir('results')

if not os.path.isfile('results/tx_2.res'):
    power = PowerLogger(interval=0.1)
    final_results = {}
    for device_id in ['cpu', 'cuda']:
        device = torch.device(device_id)
        for (data_name_, model_type_) in task_name_list:
            model, trainSet_, testSet_ = load_model_data(model_type_, data_name_)
            test_loader = DataLoader(testSet_, batch_size=1, shuffle=False)
            task_name = data_name_ + '_' + model_type_
            model.eval().to(device)
            print('--------------- %s ------------' % task_name)
            result_list = []
            for i, (x, y) in tqdm(enumerate(test_loader)):
                if i > max_test_num:
                    break
                x = x.to(device)
                is_success, sum_flops, sum_t, energy = get_overhead(repeat_num, x, model, device)
                if is_success:
                    result_list.append([sum_flops, sum_t, energy])
            final_results[task_name + '_' + device_id] = result_list
            print('----------------------------')
    for k in final_results:
        print(k, final_results[k])
    torch.save(final_results, 'results/tx_2.res')
else:
    results = torch.load('results/tx_2.res')
    final_res = []
    for k in results:
        data_set_name, model_name, device = k.split('_')
        data = results[k]
        flops, latency, energy = [], [], []
        for d in data:
            flops.append(d[0]/1000000/repeat_num)
            latency.append(d[1] / repeat_num)
            energy.append(d[2] / repeat_num)
        flops = np.array(flops)
        latency = np.array(latency)
        energy = np.array(energy)

        correct_id = np.where((energy < 1000) * (energy > 0))[0]
        flops = flops[correct_id]
        latency = latency[correct_id]
        energy = energy[correct_id]

        min_flops, avg_flops, std_flops, max_flops = np.min(flops), np.mean(flops), np.std(flops), np.max(flops)
        min_l, avg_l, std_l, max_l = np.min(latency), np.mean(latency), np.std(latency), np.max(latency)
        min_e, avg_e, std_e, max_e = np.min(energy), np.mean(energy), np.std(energy), np.max(energy)
        row = [
            k,
            min_flops, avg_flops, std_flops, max_flops,
            min_l, avg_l, std_l, max_l,
            min_e, avg_e, std_e, max_e
        ]
        final_res.append(row)
    if not os.path.isdir('final_res'):
        os.mkdir('final_res')
    with open('final_res/subject.csv', 'w') as f:
        writer = csv.writer(f)
        name = [
            'task', 'min_f', 'avg_f', 'std_f', 'max_f',
            'min_l', 'avg_l', 'std_l', 'max_l',
            'min_e', 'avg_e', 'std_e', 'max_e'
        ]
        writer.writerow(name)
        writer.writerows(final_res)
        print('successful')

exit(0)
