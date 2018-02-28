#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=0
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=base_101_EM_v1 \
--debug_mode=False \
--route=EM \
--dataset=cifar10 \
--cap_model=v0 \
--loss_form=spread \
--less_data_aug


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
