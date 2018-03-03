#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=4,5,6,7
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=cap_102_v4_new \
--cap_model=v2 \
--net_config=default_super \
--loss_fac=100 \
--schedule 300 400 450 \
--max_epoch=600 \
--debug_mode=False \
--less_data_aug \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
