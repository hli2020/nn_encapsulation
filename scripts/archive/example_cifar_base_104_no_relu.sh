#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=2 python holly_cifar.py \
--experiment_name=cifar_base_104_no_relu \
--dataset=cifar \
--model_cifar=capsule \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.01 \
--route_num=4 \
--train_batch=128 \
--test_batch=128 \
--w_version=v2 \
--b_init=zero \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
