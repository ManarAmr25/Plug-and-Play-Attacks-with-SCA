


CUDA_VISIBLE_DEVICES=2 python3 attack.py -c=configs/attacking/celeba/cnn-nod_CelebA.yaml 2>&1 | tee logs/celeba-attack/cnn-nod-attack.log
CUDA_VISIBLE_DEVICES=2 python3 attack.py -c=configs/attacking/celeba/resnet-nod_CelebA.yaml 2>&1 | tee logs/celeba-attack/resnet-nod-attack.log
CUDA_VISIBLE_DEVICES=2 python3 attack.py -c=configs/attacking/celeba/resnet18-nod_CelebA.yaml 2>&1 | tee logs/celeba-attack/resnet18-nod-attack.log
CUDA_VISIBLE_DEVICES=2 python3 attack.py -c=configs/attacking/celeba/vgg-nod_CelebA.yaml 2>&1 | tee logs/celeba-attack/vgg-nod-attack.log

CUDA_VISIBLE_DEVICES=2 python3 attack.py -c=configs/attacking/celeba/cnn-sca_CelebA.yaml 2>&1 | tee logs/celeba-attack/cnn-sca-attack.log
CUDA_VISIBLE_DEVICES=2 python3 attack.py -c=configs/attacking/celeba/resnet-sca_CelebA.yaml 2>&1 | tee logs/celeba-attack/resnet-sca-attack.log
CUDA_VISIBLE_DEVICES=2 python3 attack.py -c=configs/attacking/celeba/vgg-sca_CelebA.yaml 2>&1 | tee logs/celeba-attack/vgg-sca-attack.log
CUDA_VISIBLE_DEVICES=2 python3 attack.py -c=configs/attacking/celeba/resnet18-sca_CelebA.yaml 2>&1 | tee logs/celeba-attack/resnet18-sca-attack.log