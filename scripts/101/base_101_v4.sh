#!/bin/bash

start=`date +%s`

#--experiment_name=base_101_v4_rerun \
# _v4_less_aug
# _v4_1gpu_optimized
#
# train and test
CUDA_VISIBLE_DEVICES=3 python holly_capsule.py \
--experiment_name=base_101_v4_large_bs \
--debug_mode=False \
--dataset=cifar10 \
--cap_model=v0 \
--max_epoch=600 \
--loss_form=margin \
--optim=adam \
--schedule 150 200 250 \
--lr=0.001 \
--batch_size_train=512 \
--batch_size_train=128 \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
