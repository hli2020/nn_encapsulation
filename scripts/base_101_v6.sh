#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=7 python holly_capsule.py \
--experiment_name=base_101_v6 \
--debug_mode=False \
--dataset=cifar10 \
--cap_model=v0 \
--max_epoch=300 \
--loss_form=margin \
--optim=adam \
--pre_ch_num=32 \
--schedule 150 200 250 \
--lr=0.0001


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
