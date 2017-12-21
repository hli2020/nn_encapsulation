#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=3 python holly_cifar.py \
--experiment_name=cifar_base_107_3 \
--dataset=cifar \
--model_cifar=capsule \
--cap_model=v1 \
--skip_pre_squash \
--skip_pre_transfer \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.01 \
--route_num=4 \
--train_batch=128 \
--test_batch=128 \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
