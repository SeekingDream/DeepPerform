import os
import torch

from measure_utils import test_flops
from utils import MODEL_PATH


for date_name in MODEL_PATH:
    for model_name in MODEL_PATH[date_name]:
        for norm in ['inf', '2']:
            if norm == 'inf':
                per_size_list = [0.03, 0.06, 0.09, 0.12, 0.15]
            else:
                per_size_list = [10, 15, 20, 25, 30]
            for per_size in per_size_list:
                task_name = model_name + '_' + date_name + '_' + norm + '_hyper0.001_' + str(per_size)
                model_path = os.path.join('AbuseGanModel', task_name)
                model_path = os.path.join(model_path, 'best.tar')
                r = torch.load(model_path)
                print()


