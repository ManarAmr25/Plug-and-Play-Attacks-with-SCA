#!/bin/bash
#SBATCH --job-name=celeb-nod-cnn
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/celeba/celeba-nod-cnn-%j.out

# source ~/.bashrc
# conda activate /home/m.saeed/miniconda3/envs/sca-pnp


CUDA_VISIBLE_DEVICES=5 python3 train_model.py -c=configs/training/targets/cifar10/resnetsca_Cifar10.yaml 2>&1 | tee logs/cifar10/cifar10-sca_resnet.log
