


CUDA_VISIBLE_DEVICES=6 python3 attack.py -c=configs/attacking/cifar10/cnn-sca_Cifar10.yaml 2>&1 | tee logs/cifar10-attack/cnn-sca-attack.log
CUDA_VISIBLE_DEVICES=6 python3 attack.py -c=configs/attacking/cifar10/resnet-sca_Cifar10.yaml 2>&1 | tee logs/cifar10-attack/resnet-sca-attack.log
CUDA_VISIBLE_DEVICES=6 python3 attack.py -c=configs/attacking/cifar10/resnet18-sca_Cifar10.yaml 2>&1 | tee logs/cifar10-attack/resnet18-sca-attack.log
CUDA_VISIBLE_DEVICES=6 python3 attack.py -c=configs/attacking/cifar10/vgg-sca_Cifar10.yaml 2>&1 | tee logs/cifar10-attack/vgg-sca-attack.log
