#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
                                 --dataset /media/mumu/00003799000C7B90/WorkSpace/super-resolution/data/CelebASubset \
                                 --imgsize 128 \
                                 --scale 4 \
                                 --globallayers 16 \
                                 --locallayers 8 \
                                 --featuresize 64 \
                                 --batchsize 10 \
                                 --savedir saved_models \
                                 --iterations 1000 \
                                 --usepre 0
