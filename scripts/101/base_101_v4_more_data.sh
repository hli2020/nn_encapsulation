#!/bin/bash

start=`date +%s`

#--experiment_name=base_101_v4_rerun \
# _v4_less_aug
# _v4_1gpu_optimized
#
# train and test
DEVICE_ID=4,5
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=base_101_v4_more_data \
--debug_mode=False \
--dataset=cifar10 \
--cap_model=v0 \
--max_epoch=600 \
--loss_form=margin \
--optim=adam \
--schedule 150 200 250 \
--lr=0.0001 \
--batch_size_train=128 \
--batch_size_test=128 \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
