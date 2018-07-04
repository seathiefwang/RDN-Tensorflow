#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
                                 --dataset data/DIV2K \
                                 --imgsize 128 \
                                 --scale 4 \
                                 --globallayers 16 \
                                 --locallayers 8 \
                                 --featuresize 64 \
                                 --batchsize 10 \
                                 --savedir saved_models \
                                 --iterations 1000 \
                                 --usepre 0
