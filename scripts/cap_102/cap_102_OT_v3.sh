#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=0
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=cap_102_OT_v3 \
--cap_model=v2 \
--net_config=set_OT \
--ot_loss \
--ot_loss_fac=20 \
--schedule 350 450 550 \
--no_bp_P_L \
--debug_mode=False \
--less_data_aug \
--s35

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
