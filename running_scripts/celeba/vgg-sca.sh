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


echo "########### vgg sca"
CUDA_VISIBLE_DEVICES=1 start_ckpt=results/celeba-vgg-sca/vggSca_20250712_211836/Classifierepoch_8_0.6435_no_val.pth python3 train_model.py -c=configs/training/targets/vggsca_CelebA.yaml 2>&1 | tee  logs/celeba/celeba-sca_vgg2.log
