

# python trainAbuseGan.py --config 'config/cifar10_blockdrop.json'


CUDA_VISIBLE_DEVICES=3 python trainAbuseGan.py --config 'config/linf/cifar10_DeepShallow_inf.json'
CUDA_VISIBLE_DEVICES=3 python trainAbuseGan.py --config 'config/linf/cifar100_DeepShallow_inf.json'


# python runILFO.py --config 'ilfo_config/cifar10_blockdrop_inf.json'
# python runILFO.py --config 'ilfo_config/cifar10_blockdrop.json'