#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=1 python holly_cifar.py \
--experiment_name=cifar_base_104_no_relu_rmsprop \
--dataset=cifar \
--model_cifar=capsule \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.001 \
--optim=rmsprop \
--route_num=4 \
--train_batch=128 \
--test_batch=128 \
--w_version=v2 \
--b_init=zero \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
