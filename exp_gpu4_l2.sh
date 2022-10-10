
CUDA_VISIBLE_DEVICES=4 python trainAbuseGan.py --config 'config/l2/cifar10_skipnet_l2.json'
CUDA_VISIBLE_DEVICES=4 python trainAbuseGan.py --config 'config/l2/cifar100_skipnet_l2.json'



#python runILFO.py --config 'ilfo_config/cifar10_skipnet_inf.json'
#python runILFO.py --config 'ilfo_config/cifar10_skipnet.json'
