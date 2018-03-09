#!/bin/bash

start=`date +%s`

# train and test
DEVICE_ID=0
CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=baseline_cifar10 \
--base_save_folder=result/paper \
--debug_mode=False \
--dataset=cifar10 \
--less_data_aug \
--cap_model=v_base \
--lr=0.0001 \
--optim=rmsprop \
--loss_form=CE \
--s35


CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=baseline_cifar100 \
--base_save_folder=result/paper \
--debug_mode=False \
--dataset=cifar100 \
--less_data_aug \
--cap_model=v_base \
--lr=0.0001 \
--optim=rmsprop \
--loss_form=CE \
--s35


CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=baseline_svhn \
--base_save_folder=result/paper \
--debug_mode=False \
--dataset=svhn \
--less_data_aug \
--cap_model=v_base \
--lr=0.0001 \
--optim=rmsprop \
--loss_form=CE \
--s35

CUDA_VISIBLE_DEVICES=$DEVICE_ID python holly_capsule.py \
--device_id=$DEVICE_ID \
--experiment_name=baseline_mnist \
--base_save_folder=result/paper \
--debug_mode=False \
--dataset=mnist \
--less_data_aug \
--cap_model=v_base \
--lr=0.0001 \
--optim=rmsprop \
--loss_form=CE \
--s35

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
