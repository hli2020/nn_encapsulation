#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=7 python holly_cifar.py \
--experiment_name=cifar_base_106_lr_small \
--dataset=cifar10 \
--model_cifar=capsule \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.01 \
--use_CE_loss \
--route_num=4 \
--multi_crop_test \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
