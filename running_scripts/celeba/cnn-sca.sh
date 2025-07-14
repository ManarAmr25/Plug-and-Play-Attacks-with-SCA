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


echo "########### sca"
CUDA_VISIBLE_DEVICES=3 start_ckpt=results/celeba-sca/convSca_20250713_105555/Classifier_epoch9_0.6788_no_val.pth python3 train_model.py -c=configs/training/targets/convsca_CelebA.yaml 2>&1 | tee logs/celeba/celeba-sca_cnn-3.log

# CUDA_VISIBLE_DEVICES=6 start_ckpt=results/celeba-resnet18-sca/resnet18Sca_20250712_211848/Classifierepoch_8_0.6360_no_val.pth python3 train_model.py -c=configs/training/targets/resnet18sca_CelebA.yaml 2>&1 | tee logs/celeba/celeba-sca_resnet18-2.log
