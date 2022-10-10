import argparse
import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import datetime
from tqdm import tqdm
import json
import time

import tensorboard_logger as tf_log
import logging

from utils import *
from src import ILFOAttack


def configure_log(log_dir, task_name):
    log_path = os.path.join(log_dir, task_name + '.log')
    tf_log_path = os.path.join(log_dir, task_name)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    tf_log.configure(tf_log_path)


def runILFOAttack(config):
    model_type, dataset_name, device, batch_size, \
    attack_norm, pert_lambda, max_iter, logDir, saveDir = load_configure(config)

    task_name = model_type + '_' + dataset_name + '_' + attack_norm
    configure_log(logDir, task_name)

    model, trainSet, testSet = load_model_data(model_type, dataset_name)
    logging.info('load model %s, load dataset %s' % (model_type, dataset_name))

    for k in config:
        logging.info(k + ":" + str(config[k]))

    model = model.to(device).train()

    test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=False)

    ilfo_attack = ILFOAttack(model, pert_lambda, float(attack_norm), device, max_iter)

    save_path = task_name + '_hyper' + str(pert_lambda) + '.adv'
    save_path = os.path.join(saveDir, save_path)

    results = []
    adv_metric = Metric()
    time_res = []
    for index, (images, labels) in enumerate(test_loader):
        if index >= 5:
            break
        images, labels = images.to(device), labels.to(device)

        t1 = time.time()
        new_images = ilfo_attack.transform(images.detach().cpu(), tf_log, index)
        t2 = time.time()
        time_res.append(t2 - t1)

        new_images = new_images.to(device)
        adv_preds, adv_masks, adv_probs = model(new_images, device)

        perturbation = new_images - images
        perturbation = torch.norm(perturbation.view(perturbation.shape[0], -1), float(attack_norm), dim=1)

        logging.info('%d : masks %.3f, perturbation %.3f' %
            (index, adv_masks.sum(1).mean(), perturbation.mean().item()))
        print('%d : masks %.3f, perturbation %.3f' %
            (index, adv_masks.sum(1).mean(), perturbation.mean().item()))

        metric = [
            (get_label(adv_preds) == labels).tolist(),
            adv_masks.sum(1).tolist(),
            perturbation.detach().cpu()
        ]
        adv_metric.update(metric, len(images))
        results.append(new_images.detach().cpu())
    torch.save([results, adv_metric], save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='ilfo_config/cifar10_skipnet.json',
                        help='configuration path')
    args = parser.parse_args()
    assert args.config[:4] == 'ilfo'
    with open(args.config, 'r') as f:
        configuration = json.load(f)
        print('----------   config  ---------------')
        for k in configuration:
            print(k, configuration[k])

    runILFOAttack(configuration)
