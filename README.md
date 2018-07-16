# RDN-Tensorflow
Reproduction of paper:[Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)

## Requirements

- python > 3.5
- tensorflow > 1.0
- scipy
- numpy
- pillow
- scipy
- tqdm

## Train

### Prepare training data

Download DIV2K training data.[DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)

### Begin to train

run `sh run_train.sh`
```
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
```

## Test
`python3 test.py --dataset [image dir]`

or

`python3 test.py --image [single image path]`

## The architecture of residual dense network (RDN)

![Global Residual Learning](http://5b0988e595225.cdn.sohucs.com/images/20180304/5be328bcac39423ba519f399950da2cc.png)

Figure 1. The architecture of residual dense network (RDN).

![Local Residual Learning](http://5b0988e595225.cdn.sohucs.com/images/20180304/bdf2296c6d884a3180e83ca71ad657cd.png)

Figure 2. Residual dense block (RDB) architecture.

## References

[jmiller656/EDSR-Tensorflow](https://github.com/jmiller656/EDSR-Tensorflow)
