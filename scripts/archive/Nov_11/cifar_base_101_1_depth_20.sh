#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=4 python holly_cifar.py \
--experiment_name=cifar_base_101_1_depth_20_fmnist \
--dataset=fmnist \
--model_cifar=resnet \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.1 \
--train_batch=128 \
--test_batch=128 \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
