


CUDA_VISIBLE_DEVICES=7 python3 attack.py -c=configs/attacking/cifar10/cnn-nod_Cifar10.yaml 2>&1 | tee logs/cifar10-attack/cnn-nod-attack.log
CUDA_VISIBLE_DEVICES=7 python3 attack.py -c=configs/attacking/cifar10/resnet-nod_Cifar10.yaml 2>&1 | tee logs/cifar10-attack/resnet-nod-attack.log
CUDA_VISIBLE_DEVICES=7 python3 attack.py -c=configs/attacking/cifar10/resnet18-nod_Cifar10.yaml 2>&1 | tee logs/cifar10-attack/resnet18-nod-attack.log
CUDA_VISIBLE_DEVICES=7 python3 attack.py -c=configs/attacking/cifar10/vgg-nod_Cifar10.yaml 2>&1 | tee logs/cifar10-attack/vgg-nod-attack.log
