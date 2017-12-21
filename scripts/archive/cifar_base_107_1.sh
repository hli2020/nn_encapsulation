#!/bin/bash

start=`date +%s`

# train and test
# original version of the new network
CUDA_VISIBLE_DEVICES=1 python holly_cifar.py \
--experiment_name=cifar_base_107_1 \
--dataset=cifar \
--model_cifar=capsule \
--cap_model=v1 \
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
