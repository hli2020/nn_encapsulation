#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=0,1
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=cap_102_v5 \
--cap_model=v2 \
--net_config=set1 \
--debug_mode=False \
--less_data_aug \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
