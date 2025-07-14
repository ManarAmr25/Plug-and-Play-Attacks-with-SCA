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


CUDA_VISIBLE_DEVICES=2 start_ckpt=results/celeba-resnet18-sca/resnet18Sca_20250712_211848/Classifierepoch_8_0.6360_no_val.pth python3 train_model.py -c=configs/training/targets/resnet18sca_CelebA.yaml 2>&1 | tee logs/celeba/celeba-sca_resnet18-2.log
