#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=0,1
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=cap_102_v1 \
--connect_detail=all \
--cap_model=v2 \
--manner=1 \
--debug_mode=False \
--less_data_aug


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
