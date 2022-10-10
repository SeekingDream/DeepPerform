import os
import torch
import time
from tqdm import tqdm
import csv

from src import Generator
from utils import load_model_data
from torch.utils.data import DataLoader

resDir = 'results'
if not os.path.isdir(resDir):
    os.mkdir(resDir)

device = torch.device(7)


def time2int(time_index):
    a, b, c = time_index.split(':')
    return int(a) * 3600 + int(b) * 60 + int(c)


for log_file in os.listdir('IlfoLog'):
    if '.log' not in log_file:
        continue
    task_name = log_file.split('.')[0]
    if task_name == 'SkipNet_CIFAR100_2' or task_name == 'BlockDrop_CIFAR10_2':
        continue

    model_name, data_set = task_name.split('_')[0], task_name.split('_')[1]

    log_file = os.path.join('IlfoLog', log_file)
    with open(log_file, 'r') as f:
        data = f.readlines()
    data = data[11:]
    data = [time2int(d.split(' ')[1]) for d in data]
    ilfo_cost = []
    for i in range(len(data) - 1):
        ilfo_cost.append(data[i + 1] - data[i])
    gan_model = 'AbuseGanModel/' + task_name + '_hyper0.0001/netG_epoch_90.pth'
    gan_model = torch.load(gan_model)
    model = Generator(3, 3)
    print(task_name, 'load successful')
    model.load_state_dict(gan_model)
    gan_model = model.to(device).eval()

    model, trainSet, testSet = load_model_data(model_name, data_set)
    model = model.to(device).eval()
    test_loader = DataLoader(testSet, batch_size=256, shuffle=False)

    gan_cost = []
    for (x, y) in tqdm(test_loader):
        x = x.to(device)
        t1 = time.time()
        x = gan_model(x) + x
        pred = model(x, device)
        t2 = time.time()
        gan_cost.append(t2 - t1)
    res = [[gan_cost[i], ilfo_cost[i]] for i in range(len(ilfo_cost))]
    with open(os.path.join(resDir, task_name + "_overhead.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gan", "ilfo"])
        writer.writerows(res)
    print(task_name, 'successful')
