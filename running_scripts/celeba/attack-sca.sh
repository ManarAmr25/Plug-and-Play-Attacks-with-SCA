


CUDA_VISIBLE_DEVICES=7 python3 attack.py -c=configs/attacking/celeba/cnn-sca_CelebA.yaml 2>&1 | tee logs/celeba-attack/cnn-sca-attack.log
CUDA_VISIBLE_DEVICES=7 python3 attack.py -c=configs/attacking/celeba/resnet-sca_CelebA.yaml 2>&1 | tee logs/celeba-attack/resnet-sca-attack.log
CUDA_VISIBLE_DEVICES=7 python3 attack.py -c=configs/attacking/celeba/vgg-sca_CelebA.yaml 2>&1 | tee logs/celeba-attack/vgg-sca-attack.log
CUDA_VISIBLE_DEVICES=7 python3 attack.py -c=configs/attacking/celeba/resnet18-sca_CelebA.yaml 2>&1 | tee logs/celeba-attack/resnet18-sca-attack.log