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


echo "########### nodef"
CUDA_VISIBLE_DEVICES=7 python3 train_model.py -c=configs/training/targets/convnod_CelebA.yaml 2>&1 | tee logs/celeba/celeba-nod_cnn.log