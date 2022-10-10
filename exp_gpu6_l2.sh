
CUDA_VISIBLE_DEVICES=6 python trainAbuseGan.py --config 'config/l2/cifar10_RANet_l2.json'
CUDA_VISIBLE_DEVICES=6 python trainAbuseGan.py --config 'config/l2/cifar100_RANet_l2.json'


# python trainAbuseGan.py --config 'config/cifar100_skipnet.json'




# python runILFO.py --config 'ilfo_config/cifar100_skipnet_inf.json'
# python runILFO.py --config 'ilfo_config/cifar100_skipnet.json'
