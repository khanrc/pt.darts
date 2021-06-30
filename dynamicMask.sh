#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p gpu-private
##SBATCH -p res-gpu-small
##SBATCH -p res-gpu-large
##SBATCH -p gpu-bigmem
#SBATCH --qos=short
#SBATCH --job-name=dartsMask
#SBATCH --gres=gpu:1
#SBATCH -o dynamicMask.out
#SBATCH --mem=72g
##SBATCH --mem=28g
##SBATCH --mem=48g
#SBATCH -t 2-0:0:0

#module unload cuda
#module load cuda/9.0
#source venv/bin/activate

rm ~/hem/perceptual/tempSave/maskclusters/*
bash bashDynamicMask.sh 100 0.9 0.25
