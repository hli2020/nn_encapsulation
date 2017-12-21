#!/bin/bash`:q

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=7 python holly_cifar.py \
--experiment_name=cifar_base_104_draw_stats_ALL \
--dataset=cifar \
--model_cifar=capsule \
--route_num=4 \
--train_batch=128 \
--test_batch=128 \
--draw_hist \
--test_only \
--deploy

CUDA_VISIBLE_DEVICES=7 python holly_cifar.py \
--experiment_name=cifar_base_104_draw_stats_ALL_non_target \
--dataset=cifar \
--model_cifar=capsule \
--route_num=4 \
--train_batch=128 \
--test_batch=128 \
--draw_hist \
--non_target_j \
--test_only \
--deploy


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
