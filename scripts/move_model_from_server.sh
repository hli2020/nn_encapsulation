#!/bin/bash

sshpass -p '123456' scp -P 2042 -r hyli@cdcgwb.ie.cuhk.edu.hk:\
/DATA/hyli/project/capsule/result/base_101_v4_rerun \
result/

cat opt_train_val_START_epoch_0_END_300.txt | grep -o "TEST, Top1_err: \([0-9]*.[0-9]*\)" | grep -o "[0-9]*\.[0-9]*" > tmp

# get loss
cat opt_train_val_START_epoch_0_END_300.txt | grep -o "Loss: \([0-9].[0-9]*\)" | grep -o "[0-9.]*" > tmp