


CUDA_VISIBLE_DEVICES=0 python trainAbuseGan.py --config 'config/linf/cifar10_skipnet_inf.json'
CUDA_VISIBLE_DEVICES=0 python trainAbuseGan.py --config 'config/linf/cifar100_skipnet_inf.json'



#python runILFO.py --config 'ilfo_config/cifar10_skipnet_inf.json'
#python runILFO.py --config 'ilfo_config/cifar10_skipnet.json'
