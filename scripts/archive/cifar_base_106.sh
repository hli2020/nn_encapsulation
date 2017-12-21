#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=0 python holly_cifar.py \
--experiment_name=cifar_base_106 \
--dataset=cifar \
--model_cifar=capsule \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.1 \
--use_CE_loss \
--route_num=4 \
--train_batch=128 \
--test_batch=128 \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
