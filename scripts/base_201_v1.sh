#!/bin/bash

start=`date +%s`

#--experiment_name=base_101_v4_rerun \
# train and test
CUDA_VISIBLE_DEVICES=4,5 python holly_capsule.py \
--experiment_name=base_201_v1 \
--debug_mode=False \
--dataset=tiny_imagenet \
--setting=top1 \
--cap_model=v0 \
--max_epoch=500 \
--loss_form=margin \
--optim=adam \
--schedule 200 300 400 \
--batch_size_train=512 \
--batch_size_test=256 \
--lr=0.0001


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
