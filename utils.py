import os
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
import platform

from src import *

if platform.platform() == 'Linux-4.9.140-tegra-aarch64-with-Ubuntu-18.04-bionic':
    DATA_PATH = '/home/nvidia/experiments/CM/Dataset'
    _MODEL_DIR_ = '/home/nvidia/experiments/CM/AbuseGan/model_weights'
    IS_TX2 = True
else:
    DATA_PATH = '/glusterfs/data/sxc180080/Project/Dataset'
    _MODEL_DIR_ = '/home/sxc180080/data/Project/AbuseGan/model_weights'
    IS_TX2 = False

DATASET = ['CIFAR10', 'CIFAR100']
MODEL_ARCH = ['BlockDrop', 'SkipNet', 'RANet', 'DeepShallow']

_DEEPSHALLOW_ = '_resnet56_sdn_sdn_training'  # resnet56, mobilenet, vgg16bn, wideresnet32_4

MODEL_PATH = {
    'CIFAR10': {
        'SkipNet': os.path.join(_MODEL_DIR_, 'resnet-110-rnn-cifar10.pth.tar'),
        'BlockDrop': os.path.join(_MODEL_DIR_, 'cv/finetuned/R110_C10_gamma_10/ckpt_E_2000_A_0.936_R_1.95E-01_S_16.93_#_469.t7'),
        'RANet': os.path.join(_MODEL_DIR_, 'RANet/cifar10/save_models/model_best.pth.tar'),
        'DeepShallow': os.path.join(_MODEL_DIR_, 'shallow-deep/')
    },
    'CIFAR100': {
        'SkipNet': os.path.join(_MODEL_DIR_, 'resnet-110-rnn-cifar100.pth.tar'),
        'BlockDrop': os.path.join(_MODEL_DIR_, 'cv/finetuned/R110_C100_gamma_5/ckpt_E_2000_A_0.737_R_-8.11E-01_S_30.21_#_3.t7'),
        'RANet': os.path.join(_MODEL_DIR_, 'RANet/cifar100/save_models/model_best.pth.tar'),
        'DeepShallow': os.path.join(_MODEL_DIR_, 'shallow-deep')
    },
    'SVHN': {
        'DeepShallow': os.path.join(_MODEL_DIR_, 'shallow-deep/')
    },
    # 'TinyImageNet': {
    #     'DeepShallow': os.path.join(_MODEL_DIR_, 'shallow-deep/')
    # },
    # 'ImageNet': {
    #     'SkipNet': os.path.join(_MODEL_DIR_, 'resnet-101-rnn-imagenet.pth.tar'),
    #     'BlockDrop': os.path.join(_MODEL_DIR_, 'cv/finetuned/R101_ImgNet_gamma_10/ckpt_E_10_A_0.768_R_-2.18E+00_S_29.74_#_19.t7')
    # }
}


class Metric(object):
    def __init__(self):
        self.k_list = ['acc', 'mask', 'perturbation']
        self.result = {
            'acc': {},
            'mask': {},
            'perturbation': {}
        }
        self.reset()

    def reset(self):
        for k in self.result:
            self.result[k]['avg'] = 0.0
            self.result[k]['sum'] = 0.0
            self.result[k]['count'] = 0.0
            self.result[k]['record'] = []

    def update(self, metrics: list, n: int):
        for i, k in enumerate(self.k_list):
            now_metric = metrics[i]
            self.result[k]['sum'] += sum(now_metric)
            self.result[k]['count'] += n
            self.result[k]['avg'] = self.result[k]['sum'] / self.result[k]['count']
            self.result[k]['record'] += now_metric

    def dump(self, name):
        dump_res = name + ':  '
        for k in self.result:
            dump_res += '%s: %.3f  ' % (k, self.result[k]['avg'])
        return dump_res


def load_data(data_name, model_name):
    if data_name == 'CIFAR10':
        if model_name == 'SkipNet':
            mean_val, std_val = (0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010)
        elif model_name == 'BlockDrop':
            mean_val, std_val = (0, 0, 0), (1, 1, 1)
        elif model_name == 'RANet':
            mean_val, std_val = (0, 0, 0), (1, 1, 1)
        elif model_name == 'DeepShallow':
            mean_val, std_val = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            raise NotImplemented
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean_val, std_val),
        ])

        dataPath = os.path.join(DATA_PATH, data_name)

        trainSet = torchvision.datasets.CIFAR10(
            root=dataPath, train=True, download=True,
            transform=transform_test
        )
        testSet = torchvision.datasets.CIFAR10(
            root=dataPath, train=False, download=True,
            transform=transform_test
        )
        return trainSet, testSet, mean_val, std_val

    elif data_name == 'CIFAR100':
        if model_name == 'SkipNet':
            mean_val, std_val = (0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010)
        elif model_name == 'BlockDrop':
            mean_val, std_val = (0, 0, 0), (1, 1, 1)
        elif model_name == 'RANet':
            mean_val, std_val = (0, 0, 0), (1, 1, 1)
        elif model_name == 'DeepShallow':
            mean_val, std_val = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        else:
            raise NotImplemented
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataPath = os.path.join(DATA_PATH, data_name)

        trainSet = torchvision.datasets.CIFAR100(
            root=dataPath, train=True, download=True,
            transform=transform_test
        )
        testSet = torchvision.datasets.CIFAR100(
            root=dataPath, train=False, download=True,
            transform=transform_test
        )
        return trainSet, testSet, mean_val, std_val

    elif data_name == 'ImageNet':
        mean_val, std_val = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #  transforms.Normalize(mean_val, std_val)
        ])
        dataPath = os.path.join(DATA_PATH, data_name)
        trainSet = torchvision.datasets.ImageFolder(dataPath + '/train/', transform_test)
        testSet = torchvision.datasets.ImageFolder(dataPath + '/val/', transform_test)
        return trainSet, testSet, mean_val, std_val

    elif data_name == 'TinyImageNet':
        if model_name == 'SkipNet':
            mean_val, std_val = (0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010)
        elif model_name == 'BlockDrop':
            mean_val, std_val = (0, 0, 0), (1, 1, 1)
        elif model_name == 'RANet':
            mean_val, std_val = (0, 0, 0), (1, 1, 1)
        elif model_name == 'DeepShallow':
            mean_val, std_val = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        else:
            raise NotImplemented
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataPath = os.path.join(DATA_PATH, 'tiny-imagenet-200')
        train_dir = os.path.join(dataPath, 'train')
        test_dir = os.path.join(dataPath, 'val')

        trainSet = torchvision.datasets.ImageFolder(
            train_dir, transform=transform_test
        )
        testSet = torchvision.datasets.ImageFolder(
            test_dir, transform=transform_test
        )
        return trainSet, testSet, mean_val, std_val

    elif data_name == 'SVHN':
        mean_val, std_val = (0, 0, 0), (1, 1, 1)
        dataPath = os.path.join(DATA_PATH, data_name)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainSet = torchvision.datasets.SVHN(
            root=dataPath, split='train', download=True,
            transform=transform_test
        )
        testSet = torchvision.datasets.SVHN(
            root=dataPath, split='test', download=True,
            transform=transform_test
        )
        return trainSet, testSet, mean_val, std_val

    else:
        raise NotImplemented


def _loadSkipNet(data_name, mean_val, std_val):
    if data_name == 'CIFAR10':
        model = cifar10_rnn_gate_rl_110()
    elif data_name == 'CIFAR100':
        model = cifar100_rnn_gate_rl_110()
    elif data_name == 'ImageNet':
        model = imagenet_rnn_gate_rl_101()
    else:
        raise NotImplemented
    model = torch.nn.DataParallel(model)
    device = torch.device('cpu')
    model_path = MODEL_PATH[data_name]['SkipNet']
    checkpoint = torch.load(model_path, map_location=device)
    if not IS_TX2:
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    model = model.cpu()
    model = SkipNet(model, mean_val, std_val)
    return model


def _loadBlockDrop(data_name, mean_val, std_val):
    if data_name == 'CIFAR10':
        rNet, agent = cifar10_blockdrop_110()
    elif data_name == 'CIFAR100':
        rNet, agent = cifar100_blockdrop_110()
    elif data_name == 'ImageNet':
        rNet, agent = imagenet_blockdrop_101()
    else:
        raise NotImplemented
    # gent.logit.weight.data.fill_(0)
    # agent.logit.bias.data.fill_(10)
    model_path = MODEL_PATH[data_name]['BlockDrop']

    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    if not IS_TX2:
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
    rNet.load_state_dict(checkpoint['resnet'])
    agent.load_state_dict(checkpoint['agent'])
    model = PolicyNet(rNet, agent, mean_val, std_val)
    return model


def _loadRaNet(data_name, mean_val, std_val):
    arg_parser = argparse.ArgumentParser(description='RANet Image classification')

    exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
    exp_group.add_argument('--resume', action='store_true', default=None,
                           help='path to latest checkpoint (default: none)')
    # exp_group.add_argument('--evalmode', default=None,
    #                        choices=['anytime', 'dynamic', 'both'],
    #                        help='which mode to evaluate')
    exp_group.add_argument('--evaluate-from', default='', type=str, metavar='PATH',
                           help='path to saved checkpoint (default: none)')
    exp_group.add_argument('--seed', default=0, type=int,
                           help='random seed')

    # model arch related
    arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
    arch_group.add_argument('--arch', type=str, default='RANet')
    arch_group.add_argument('--reduction', default=0.5, type=float,
                            metavar='C', help='compression ratio of DenseNet'
                                              ' (1 means dot\'t use compression) (default: 0.5)')
    # msdnet config
    arch_group.add_argument('--nBlocks', type=int, default=2)
    arch_group.add_argument('--nChannels', type=int, default=16)
    arch_group.add_argument('--growthRate', type=int, default=6)
    arch_group.add_argument('--grFactor', default='4-2-1-1', type=str)
    arch_group.add_argument('--bnFactor', default='4-2-1-1', type=str)
    arch_group.add_argument('--block-step', type=int, default=2)
    arch_group.add_argument('--scale-list', default='1-2-3-3', type=str)
    arch_group.add_argument('--compress-factor', default=0.25, type=float)
    arch_group.add_argument('--step', type=int, default=4)
    arch_group.add_argument('--stepmode', type=str, default='even', choices=['even', 'lg'])
    arch_group.add_argument('--bnAfter', action='store_true', default=True)
    arch_group.add_argument('--data', type=str, default=data_name.lower())
    arch_group.add_argument('--config', type=str, default=None)
    args = arg_parser.parse_args()

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.scale_list = list(map(int, args.scale_list.split('-')))
    args.nScales = len(args.grFactor)

    model = RANet(args)
    model = torch.nn.DataParallel(model)
    device = torch.device('cpu')
    model_path = MODEL_PATH[data_name]['RANet']
    checkpoint = torch.load(model_path, map_location=device)
    if not IS_TX2:
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    model = RasPolicyNet(model, mean_val, std_val)
    return model


def _loadDeepShallow(data_name, mean_val, std_val):
    model_path = MODEL_PATH[data_name]['DeepShallow']
    if data_name != 'SVHN':
        model_name = data_name.lower() + _DEEPSHALLOW_
    else:
        model_name = data_name.lower() + '_mobilenet_sdn_sdn_training'

    sdn_model, _ = load_deepshallow_model(model_path, model_name, epoch=-1)
    model = DeepShallowPolicyNet(sdn_model, mean_val, std_val)
    return model


def load_model_data(model_type, data_name):
    trainSet, testSet, mean_val, std_val = load_data(data_name, model_type)
    print('Load   ', data_name, '    Successful')
    if model_type == 'SkipNet':
        model = _loadSkipNet(data_name, mean_val, std_val)
        print('Load SkipNet Successful')
    elif model_type == 'BlockDrop':
        model = _loadBlockDrop(data_name, mean_val, std_val)
        print('Load BlockDrop Successful')
    elif model_type == 'RANet':
        model = _loadRaNet(data_name, mean_val, std_val)
        print('Load RaNet Successful')
    elif model_type == 'DeepShallow':
        model = _loadDeepShallow(data_name, mean_val, std_val)
        print('Load DeepShallow Successful')
    else:
        raise NotImplemented
    return model, trainSet, testSet


def get_label(pred):
    return pred.max(dim=1)[1]


@torch.no_grad()
def test_performance(data_loader, model, device, norm, defense):
    model.eval()
    metrics = Metric()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        if defense is not None:
            try:
                new_images = defense.transform(images.detach().cpu())
            except:
                new_images = defense.transform(images.detach())
        else:
            new_images = images
        new_images = new_images.to(device)
        ori_preds, ori_masks, ori_probs = model(new_images, device)
        perturbation = new_images - images
        perturbation = torch.norm(perturbation.view(perturbation.shape[0], -1), norm, dim=1)
        metric = [
            (get_label(ori_preds).cpu() == labels.cpu()).tolist(),
            ori_masks.mean(1).tolist(),
            perturbation.detach().cpu()
        ]
        metrics.update(metric, len(images))
    model.train()
    return metrics


def load_configure(config):
    model_type, dataset_name = config['model'], config['dataset']
    device = torch.device(config['device'])
    batch_size = config['batch']
    max_iter = config['max_iter']
    attack_norm = config['norm']
    pert_lambda = config['pert_lambda']
    return model_type, dataset_name, device, batch_size, attack_norm, pert_lambda, max_iter


def test_func():
    from torch.utils.data import DataLoader
    if not os.path.isdir('results'):
        os.mkdir('results')
    device = torch.device(0)
    results = {}

    model_, trainSet_, testSet_ = load_model_data('DeepShallow', 'SVHN')

    for data_name_ in MODEL_PATH:
        for model_type_ in MODEL_PATH[data_name_]:
            model_, trainSet_, testSet_ = load_model_data(model_type_, data_name_)
            # val_loader = DataLoader(trainSet_, batch_size=200)
            test_loader = DataLoader(testSet_, batch_size=1)
            task_name = data_name_ + '_' + model_type_
            model_.eval().to(device)
            ops_list = []
            print('--------------- %s ------------', task_name)
            for i, (x, y) in enumerate(test_loader):
                if i > 1:
                    break
                x = x.to(device)
                masks, ops = model_.adaptive_forward(x, device)
                ops_list.append(ops)
                print(ops)
            results[task_name] = ops_list
            print('----------------------------')


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    test_func()
