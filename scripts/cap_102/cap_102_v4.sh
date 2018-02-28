#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=3
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=cap_102_v4_1gpu \
--connect_detail=all \
--cap_model=v2 \
--manner=3 \
--debug_mode=False \
--less_data_aug \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
