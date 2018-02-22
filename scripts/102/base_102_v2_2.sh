#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=5
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=base_102_v2_2 \
--connect_detail=all \
--more_skip \
--layerwise \
--cap_model=v2 \
--debug_mode=False \
--less_data_aug \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
