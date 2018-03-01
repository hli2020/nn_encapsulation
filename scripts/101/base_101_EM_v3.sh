#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=0
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=base_101_EM_v3_long_phase1 \
--debug_mode=False \
--route=EM \
--schedule 400 500 600 \
--max_epoch=700 \
--dataset=cifar10 \
--cap_model=v0 \
--loss_form=margin \
--less_data_aug


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"