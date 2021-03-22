#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p gpu-private
##SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=nas
#SBATCH --gres=gpu:1
#SBATCH -o outputSearch.out
#SBATCH --mem=72g
##SBATCH --mem=28g
#SBATCH -t 2-0:0:0

#module unload cuda
#module load cuda/9.0
#source venv/bin/activate

#python3 augment.py --name cifar10-mg --dataset cifar10 --gpus 0 \
#    --batch_size 32 --workers 8 --print_freq 50 --lr 0.1 \
#    --genotype "Genotype(
#    normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], [('skip_connect', 0), ('dil_conv_3x3', 2)], [('sep_conv_3x3', 1), ('skip_connect', 0)], [('sep_conv_3x3', 1), ('skip_connect', 0)]],
#    normal_concat=range(2, 6),
#    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('skip_connect', 3), ('max_pool_3x3', 0)], [('skip_connect', 2), ('max_pool_3x3', 0)]],
#    reduce_concat=range(2, 6)
#)"

bash bashSearch.sh

