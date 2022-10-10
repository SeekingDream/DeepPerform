

# python trainAbuseGan.py --config 'config/cifar10_blockdrop.json'


CUDA_VISIBLE_DEVICES=7 python trainAbuseGan.py --config 'config/l2/cifar10_DeepShallow_l2.json'
CUDA_VISIBLE_DEVICES=7 python trainAbuseGan.py --config 'config/l2/cifar100_DeepShallow_l2.json'


# python runILFO.py --config 'ilfo_config/cifar10_blockdrop_inf.json'
# python runILFO.py --config 'ilfo_config/cifar10_blockdrop.json'