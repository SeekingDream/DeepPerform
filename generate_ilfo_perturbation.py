import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from utils import *
from src import ILFOAttack
import tensorboard_logger as tf_log

parser = argparse.ArgumentParser()
parser.add_argument('--norm', default='inf', type=str)
args = parser.parse_args()


ganDir = "AbuseGanPerturbation/"
saveDir = 'ILFOPerturbation/'
logDir = 'ILFOPerturbation/tf_log'

if not os.path.isdir(saveDir):
    os.mkdir(saveDir)
if not os.path.isdir(logDir):
    os.mkdir(logDir)
tf_log.configure(logDir)


def run_ilfo(model_name, dataset_name, attack_norm):
    device = torch.device('cuda')
    pert_lambda = 1000000
    task_name = 'general_' + model_name + '_' + dataset_name + '_' + str(attack_norm)
    save_sub_dir = os.path.join(saveDir, task_name)

    if not os.path.isdir(save_sub_dir):
        os.mkdir(save_sub_dir)

    model, trainSet, testSet = load_model_data(model_name, dataset_name)
    ilfo_attack = ILFOAttack(model, pert_lambda, float(attack_norm), device, max_iter=300)
    ilfo_attack.model = ilfo_attack.model.to(device)
    test_loader = DataLoader(testSet, batch_size=200, shuffle=False)
    imgs, generated_per, overhead = [], [], []
    for i, (x, y) in tqdm(enumerate(test_loader)):
        if i >= 5:
            break
        imgs.append(x.detach().cpu())
        x = x.to(device)
        t1 = time.time()
        new_x = ilfo_attack.transform(x.detach().cpu(), tf_log=tf_log, index=i, task_name=task_name)
        t2 = time.time()
        overhead.append(t2 - t1)
        generated_per.append(new_x.detach().cpu())
    imgs = torch.cat(imgs)
    generated_per = torch.cat(generated_per)
    per_path = os.path.join(save_sub_dir, 'perturbation.per')
    latency_path = os.path.join(save_sub_dir, 'overhead.tar')
    torch.save(generated_per, per_path, _use_new_zipfile_serialization=False)
    torch.save(overhead, latency_path)
    print(task_name, 'successful')
    return imgs, generated_per


def main():
    for dataset_name in MODEL_PATH:
        if dataset_name != 'SVHN':    #todo
            continue
        for model_name in MODEL_PATH[dataset_name]:
            for attack_norm in ['2', 'inf']:
                if attack_norm == 'inf':
                    per_size_list = [0.03, 0.06, 0.09, 0.12, 0.15]
                else:
                    per_size_list = [10, 15, 20, 25, 30]
                attack_norm = float(attack_norm)

                task_name = 'general_' + model_name + '_' + dataset_name + '_' + str(attack_norm)
                save_sub_dir = os.path.join(saveDir, task_name)
                per_path = os.path.join(save_sub_dir, 'perturbation.per')
                if os.path.isfile(per_path):
                    perturbation = torch.load(per_path)
                    _, _, testSet = load_model_data(model_name, dataset_name)
                    test_loader = DataLoader(testSet, batch_size=200, shuffle=False)
                    imgs = []
                    for i, (x, y) in tqdm(enumerate(test_loader)):
                        if i >= 5:
                            break
                        imgs.append(x.detach().cpu())
                    imgs = torch.cat(imgs)
                else:
                    imgs, perturbation = run_ilfo(model_name, dataset_name, attack_norm)

                for per_size in per_size_list:
                    if attack_norm == float('inf'):
                        images = torch.clamp(perturbation, per_size, per_size) + imgs
                    else:
                        ori_shape = perturbation.shape
                        per_norm = perturbation.reshape([len(imgs), -1]).norm(p=attack_norm, dim=-1)
                        per_norm = per_norm.reshape([len(imgs), -1])
                        perturbation = perturbation.reshape([len(imgs), -1]) / per_norm * per_size
                        perturbation = perturbation.reshape(ori_shape)
                        images = perturbation + imgs
                    images = torch.clamp(images, 0, 1).detach()
                    task_name = model_name + '_' + dataset_name + '_' + str(attack_norm) + '_' + str(per_size)
                    save_sub_dir = os.path.join(saveDir, task_name)
                    if not os.path.isdir(save_sub_dir):
                        os.mkdir(save_sub_dir)
                    per_path = os.path.join(save_sub_dir, 'perturbation.per')
                    torch.save(images, per_path, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
