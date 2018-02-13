#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=0
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=base_102_v1_2_fc \
--fc_manner=fc \
--debug_mode=False \
--dataset=cifar10 \
--less_data_aug \
--cap_model=v1_2 \
--max_epoch=600 \
--loss_form=margin \
--optim=adam \
--schedule 200 300 400 \
--lr=0.0001 \
--batch_size_train=128 \
--batch_size_test=128 \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
