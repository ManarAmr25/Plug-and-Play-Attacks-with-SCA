echo "########### nodef"
# CUDA_VISIBLE_DEVICES=4 python3 train_model.py -c=configs/training/targets/cifar10/convnod_Cifar10.yaml 2>&1 | tee logs/cifar10/cifar10-conv-nod.log
CUDA_VISIBLE_DEVICES=4 python3 train_model.py -c=configs/training/targets/cifar10/resnetnod_Cifar10.yaml 2>&1 | tee logs/cifar10/cifar10-resnet-nod.log
CUDA_VISIBLE_DEVICES=4 python3 train_model.py -c=configs/training/targets/cifar10/resnet18nod_Cifar10.yaml 2>&1 | tee logs/cifar10/cifar10-resnet18-nod.log
CUDA_VISIBLE_DEVICES=4 python3 train_model.py -c=configs/training/targets/cifar10/vggnod_Cifar10.yaml 2>&1 | tee logs/cifar10/cifar10-vgg-nod.log
