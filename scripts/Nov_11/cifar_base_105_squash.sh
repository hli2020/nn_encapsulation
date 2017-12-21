#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=5 python holly_cifar.py \
--experiment_name=cifar_base_105_squash_cifar100 \
--dataset=cifar100 \
--model_cifar=capsule \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.01 \
--train_batch=128 \
--test_batch=128 \
--do_squash \
--w_version=v3 \
--multi_crop_test \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
