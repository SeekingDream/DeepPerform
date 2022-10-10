
CUDA_VISIBLE_DEVICES=2 python trainAbuseGan.py --config 'config/linf/cifar10_RANet_inf.json'
CUDA_VISIBLE_DEVICES=2 python trainAbuseGan.py --config 'config/linf/cifar100_RANet_inf.json'


# python trainAbuseGan.py --config 'config/cifar100_skipnet.json'




# python runILFO.py --config 'ilfo_config/cifar100_skipnet_inf.json'
# python runILFO.py --config 'ilfo_config/cifar100_skipnet.json'
