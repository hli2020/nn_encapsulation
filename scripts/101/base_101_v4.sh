#!/bin/bash

start=`date +%s`

#--experiment_name=base_101_v4_rerun \
# _v4_less_aug
# _v4_1gpu_optimized
#
# train and test
DEVICE_ID=2,3
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=base_101_v4_less_aug_rerun_2 \
--debug_mode=False \
--dataset=cifar10 \
--less_data_aug \
--cap_model=v0 \
--max_epoch=600 \
--loss_form=margin \
--optim=adam \
--schedule 150 200 250 \
--lr=0.0001 \
--batch_size_train=512 \
--batch_size_test=128 \
--show_test_after_epoch=10 \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
