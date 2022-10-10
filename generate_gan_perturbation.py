import os
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from utils import *

ganDir = "AbuseGanModel/"
saveDir = 'AbuseGanPerturbation/'

if not os.path.isdir(saveDir):
    os.mkdir(saveDir)

device = torch.device(5)

for date_name in MODEL_PATH:
    for model_name in MODEL_PATH[date_name]:
        model, trainSet, testSet = load_model_data(model_name, date_name)
        test_loader = DataLoader(testSet, batch_size=100, shuffle=False)
        for norm in ['inf', '2']:
            if norm == 'inf':
                per_size_list = [0.03, 0.06, 0.09, 0.12, 0.15]
            else:
                per_size_list = [10, 15, 20, 25, 30]
            for per_size in per_size_list:
                task_name = model_name + '_' + date_name + '_' + norm + '_' + str(per_size)
                save_sub_dir = os.path.join(saveDir, task_name)
                if os.path.isfile(os.path.join(save_sub_dir, 'perturbation.per')):
                    continue


                task_name = model_name + '_' + date_name + '_' + norm + '_hyper0.001_' + str(per_size)
                model_path = os.path.join('AbuseGanModel', task_name)
                model_path = os.path.join(model_path, 'best.tar')
                gan, _ = torch.load(model_path, map_location=torch.device('cpu'))
                gan.netG = gan.netG.eval().to(device)

                generated_x, overhead = [], []
                st_time = time.time()
                for i, (x, y) in enumerate(test_loader):
                    if i >= 10:
                        break
                    x = x.to(device)
                    t1 = time.time()
                    new_x = gan.transform(x)
                    t2 = time.time()
                    overhead.append(t2 - t1)
                    generated_x.append(new_x.detach().cpu())
                generated_x = torch.cat(generated_x)
                ed_time = time.time()

                task_name = model_name + '_' + date_name + '_' + norm + '_' + str(per_size)
                save_sub_dir = os.path.join(saveDir, task_name)
                if not os.path.isdir(save_sub_dir):
                    os.mkdir(save_sub_dir)
                per_path = os.path.join(save_sub_dir, 'perturbation.per')
                latency_path = os.path.join(save_sub_dir, 'overhead.tar')
                t1 = time.time()
                torch.save(generated_x, per_path, _use_new_zipfile_serialization=False)
                torch.save(overhead, latency_path)
                t2 = time.time()
                print(task_name, 'successful', 'generate cost: ', ed_time - st_time, 'save cost time: ', t2 - t1)


