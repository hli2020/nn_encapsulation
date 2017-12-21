#!/bin/bash

start=`date +%s`

# with new data augmentation and lr scheduler
# train and test
CUDA_VISIBLE_DEVICES=1 python holly_cifar.py \
--experiment_name=cifar_base_104_NEW_2 \
--dataset=cifar \
--model_cifar=capsule \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.01 \
--route_num=4 \
--train_batch=128 \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
