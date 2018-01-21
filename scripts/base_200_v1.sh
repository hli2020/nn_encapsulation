#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=0,1 python holly_capsule.py \
--experiment_name=base_200_v1 \
--debug_mode=False \
--dataset=tiny_imagenet \
--setting=top1 \
--cap_model=v_base \
--depth=14 \
--max_epoch=500 \
--schedule 200 300 400 \
--batch_size_train=512 \
--batch_size_test=256 \
--lr=0.0001 \


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
