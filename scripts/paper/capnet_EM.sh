#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=1
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=capnet_EM \
--base_save_folder=result/paper \
--debug_mode=False \
--dataset=cifar10 \
--less_data_aug \
--net_config=capnet_default \
--route=EM \
--lr=0.0001 \
--batch_size_train=2 \
--optim=adam \
--loss_form=margin \


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
