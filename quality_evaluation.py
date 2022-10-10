from math import log10, sqrt
import numpy as np
import torch
from torch.utils.data import DataLoader
import csv


from utils import *


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


GanPath = {
    '2': {
        'CIFAR10': {
            'SkipNet': 'SkipNet_CIFAR10_2_hyper0.0001/best.tar',
            'BlockDrop': 'BlockDrop_CIFAR10_2_hyper1e-05/best.tar'
        },
        'CIFAR100': {
            'SkipNet': 'SkipNet_CIFAR100_2_hyper1e-08/best.tar',
            'BlockDrop': 'BlockDrop_CIFAR100_2_hyper1e-08/best.tar'
        }
    }
}


def main():
    device = torch.device(1)
    for file_name in os.listdir('IlfoRes'):
        exp_setting = file_name.split('_')
        model_type, dataset, attack_norm = exp_setting[:3]
        file_name = os.path.join('IlfoRes', file_name)
        if attack_norm in GanPath:
            gan_path = GanPath[attack_norm][dataset][model_type]
            if gan_path == 'SkipNet_CIFAR100_2_hyper1e-08':
                print()
            if gan_path is None:
                continue
            gan_path = 'AbuseGanModel/' + gan_path
        else:
            continue

        ilfo_res = torch.load(file_name)
        ilfo_image = ilfo_res[0]
        model, _, testSet = load_model_data(model_type, dataset)
        model = model.to(device).eval()
        test_loader = DataLoader(testSet, batch_size=100, shuffle=False)

        check_point = torch.load(gan_path, map_location=device)
        gan_model = check_point[0]
        gan_model.device = device

        # gan_model = Generator(3, 3)
        # gan_model.load_state_dict(check_point)
        # gan_model = gan_model.eval().to(device)

        ilfo_metric, gan_metric = [], []
        ori_block, ilfo_block, gan_block = [], [], []
        print(file_name)
        for i, (xs, ys) in enumerate(test_loader):
            if i >= 5:
                break
            ilfo_xs = ilfo_image[i].to(device)
            xs = xs.to(device)
            gan_xs = gan_model.transform(xs.detach()) #torch.clamp(gan_model(xs) + xs, 0, 1)

            _, ori_masks, _ = model(xs, device)
            _, ilfo_masks, _ = model(ilfo_xs, device)
            _, gan_masks, _ = model(gan_xs, device)

            ori_block.extend(ori_masks.sum(1).tolist())
            ilfo_block.extend(ilfo_masks.sum(1).tolist())
            gan_block.extend(gan_masks.sum(1).tolist())

            xs = xs.detach().cpu()
            gan_xs = gan_xs.detach().cpu()
            ilfo_xs = ilfo_xs.detach().cpu()
            for ori_x, ilfo_x, gan_x in zip(xs, ilfo_xs, gan_xs):

                ori_x = np.int32(ori_x.numpy() * 255.0)
                ilfo_x = np.int32(ilfo_x.numpy() * 255.0)
                gan_x = np.int32(gan_x.numpy() * 255.0)

                ilfo_metric.append(PSNR(ori_x, ilfo_x))
                gan_metric.append(PSNR(ori_x, gan_x))
        assert len(ilfo_metric) == 500
        print("%s, %.3f, %.3f" %
              (file_name, sum(ilfo_metric) / len(ilfo_metric), sum(gan_metric)/len(gan_metric)))

        ilfo_inc = [(y / x - 1) for (x, y) in zip(ori_block, ilfo_block)]
        gan_inc = [(y / x - 1) for (x, y) in zip(ori_block, gan_block)]

        print("ILFO, min :%.3f, avg :%.3f, max :%.3f,"
              % (min(ilfo_inc), sum(ilfo_block)/sum(ori_block) - 1, max(ilfo_inc) ))

        print("Gan, min :%.3f, avg :%.3f, max :%.3f,"
              % (min(gan_inc), sum(gan_block)/sum(ori_block) - 1, max(gan_inc)))

        res = [[ilfo_metric[i], gan_metric[i]] for i in range(len(ilfo_metric))]
        save_table_name = 'results/quality' + model_type + '_' + dataset + '_' + attack_norm + '.csv'
        with open(save_table_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["gan", "ilfo"])
            writer.writerows(res)
        print(save_table_name, 'successful')


main()