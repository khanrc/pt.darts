#!/bin/bash

source ~/nas/bin/activate

SECONDS=0
python3 search.py --name cifar10-mg --dataset cifar10 --gpus 0 \
    --batch_size 8 --workers 1 --layers 4 --print_freq 10 \
    --w_lr 0.01 --w_lr_min 0.004 --alpha_lr 0.0012 --w_grad_clip 3
echo "time is $SECONDS"
