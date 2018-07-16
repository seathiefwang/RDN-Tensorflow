#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
                                 --dataset DIV2k  \
                                 --testset '' \
                                 --imgsize 156 \
                                 --scale 4 \
                                 --globallayers 8 \
                                 --locallayers 4 \
                                 --featuresize 64 \
                                 --batchsize 16 \
                                 --savedir saved_models \
                                 --iterations 100000 \
                                 --usepre 0
