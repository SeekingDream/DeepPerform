import torch
import torch.nn as nn
import os
import numpy as np


def count_flops(model, reset=True):
    op_count = 0
    for m in model.modules():
        if hasattr(m, 'num_ops'):
            op_count += m.num_ops
            if reset:         # count and reset to 0
                m.num_ops = 0
    return op_count


def modify_weights(net, mean_val, std_val):
    (name, conv) = list(net.named_children())[0]
    old_weight = conv.weight.data.detach()
    is_bias = (conv.bias is None)
    conv = torch.nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        padding_mode=conv.padding_mode
    )
    in_dim = conv.in_channels
    out_dim = conv.out_channels

    w = conv.weight = torch.nn.Parameter(old_weight, requires_grad=False)
    for i in range(in_dim):
        w[:, i, :, :] = w[:, i, :, :] / std_val[i]
    if is_bias:
        conv.bias = torch.nn.Parameter(torch.zeros([out_dim]), requires_grad=False)
    else:
        conv.bias = conv.bias
    bias = conv.bias
    for i in range(out_dim):
        for j in range(in_dim):
            bias[i] -= mean_val[j] * w[i, j, :, :].sum()
    net.add_module(name, conv)
    return net


class SkipNet(torch.nn.Module):
    '''
    wrapper class for skipnet
    '''

    def __init__(self, net, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.net = modify_weights(net, mean, std)
        self.flops = None

    def forward(self, x, device):
        preds, masks, probs = self.net(x, device)
        masks = torch.stack(masks).T
        probs = torch.stack(probs)
        probs = probs[:, :, 1]
        probs = probs.permute(1, 0)
        return preds, masks, probs

    def adaptive_forward(self, x, device):

        _, masks, _ = self.net.adaptive_forward(x, device)
        masks = torch.stack(masks).T

        ops = count_flops(self)
        return masks, ops


class PolicyNet(torch.nn.Module):
    '''
    wrapper class for blockDrop
    '''

    def __init__(self, rnet, agent, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.rnet = self.modify_weights(rnet)
        self.agent = self.modify_weights(agent)

    def modify_weights(self, net):
        (name, conv) = list(net.named_children())[0]
        if 'conv' not in name:
            conv = self.modify_weights(conv)
        else:
            is_bias = (conv.bias is None)
            old_weight = conv.weight.data.detach()
            conv = torch.nn.Conv2d(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                padding_mode=conv.padding_mode)
            in_dim = conv.in_channels
            out_dim = conv.out_channels

            w = conv.weight = torch.nn.Parameter(old_weight, requires_grad=False)
            for i in range(in_dim):
                w[:, i, :, :] = w[:, i, :, :] / self.std[i]
            if is_bias:
                conv.bias = torch.nn.Parameter(torch.zeros([out_dim]), requires_grad=False)
            else:
                conv.bias = conv.bias
            bias = conv.bias
            for i in range(out_dim):
                for j in range(in_dim):
                    bias[i] -= self.mean[j] * w[i, j, :, :].sum()
        net.add_module(name, conv)
        return net

    def forward(self, x, device):
        probs, _ = self.agent(x, device)
        policy = probs.detach().clone()
        policy[policy < 0.5] = 0.0
        policy[policy >= 0.5] = 1.0
        policy = policy.data.squeeze(0)
        preds = self.rnet.forward_single(x, probs, device)
        return preds, policy, probs

    def adaptive_forward(self, x, device):
        probs, _ = self.agent(x, device)
        policy = probs.detach().clone()
        policy[policy < 0.5] = 0.0
        policy[policy >= 0.5] = 1.0
        policy = policy.data.squeeze(0)
        self.rnet.adaptive_forward(x, probs, device)

        ops = count_flops(self)
        return policy, ops


class RasPolicyNet(torch.nn.Module):
    '''
    wrapper class for RaNet
    '''
    def __init__(self, model, mean, std):
        super(RasPolicyNet, self).__init__()
        self.model = model
        self.mean = mean
        self.std = std
        self.threshold = [
            0.5,  0.5,  0.5,  0.5,
            0.5,  0.5,  0.5, -1.0000e+08
        ]
        # self.flops = [
        #     15814410.0, 31283220.0, 44655390.0, 50358312.0,
        #     60654578.0, 63647996.0, 90173958.0, 94904592.0
        # ]

    def forward(self, x, device):
        x = x.to(device)
        logits = self.model(x)
        n_stage, n_sample = len(logits), len(logits[0])
        logits = torch.stack(logits)
        probs = nn.functional.softmax(logits, dim=2)
        max_preds, argmax_preds = probs.max(dim=2, keepdim=False)

        policy = torch.zeros([n_sample, n_stage])
        preds = []
        for i in range(n_sample):
            for k in range(n_stage):
                if max_preds[k][i].item() >= self.threshold[k]:  # force to exit at k
                    _pred = logits[k][i].detach().cpu()
                    preds.append(_pred)
                    policy[i, : k + 1] = 1
                    break
        max_preds = 1 - max_preds.T
        max_preds = (1 - policy.to(max_preds.device)) + max_preds * policy.to(max_preds.device)
        return torch.stack(preds), policy, max_preds

    def adaptive_forward(self, x, device):
        x = x.to(device)
        masks = self.model.adaptive_forward(x, self.threshold)
        ops = count_flops(self)
        return masks, ops


class DeepShallowPolicyNet(torch.nn.Module):
    '''
    wrapper class for DeepShallowPolicyNet
    '''
    def __init__(self, model, mean, std):
        super(DeepShallowPolicyNet, self).__init__()
        self.threshold = 0.5
        model.confidence_threshold = self.threshold
        self.model = self.modify_net(model, mean, std)
        self.mean = mean
        self.std = std

    @staticmethod
    def modify_net(net, mean_val, std_val):
        ori_conv = list(net.init_conv.named_children())
        name, conv = ori_conv[0]
        old_weight = conv.weight.data.detach()
        is_bias = (conv.bias is None)
        conv = torch.nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            padding_mode=conv.padding_mode
        )
        in_dim = conv.in_channels
        out_dim = conv.out_channels

        w = conv.weight = torch.nn.Parameter(old_weight, requires_grad=False)
        for i in range(in_dim):
            w[:, i, :, :] = w[:, i, :, :] / std_val[i]
        if is_bias:
            conv.bias = torch.nn.Parameter(torch.zeros([out_dim]), requires_grad=False)
        else:
            conv.bias = conv.bias
        bias = conv.bias
        for i in range(out_dim):
            for j in range(in_dim):
                bias[i] -= mean_val[j] * w[i, j, :, :].sum()
        net.init_conv.add_module(name, conv)
        return net

    def forward(self, x, device):
        x = x.to(device)
        logits = self.model.adaptive_forward(x)
        logits = torch.stack(logits)
        policy = torch.zeros([len(x), len(logits)])
        confidences = logits.softmax(-1)
        max_confidence = confidences.max(-1)[0]
        preds = []
        for i in range(len(x)):
            is_find = False
            for j in range(len(logits) - 1):
                if max_confidence[j][i] > self.threshold:
                    policy[i][:j + 1] = 1
                    preds.append(logits[j][i])
                    is_find = True
                    break
            if not is_find:
                preds.append(logits[-1][i])
                policy[i] = 1

        max_confidence = 1 - max_confidence.T

        max_confidence = (1 - policy.to(max_confidence.device)) + max_confidence * policy.to(max_confidence.device)
        return torch.stack(preds), policy, max_confidence

    def adaptive_forward(self, x, device):
        x = x.to(device)
        masks = self.model.early_exit(x, self.threshold)

        ops = count_flops(self)
        return masks, ops
