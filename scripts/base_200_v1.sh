#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=5,6 python holly_capsule.py \
--experiment_name=base_200_v1_try1 \
--num_workers=8 \
--debug_mode=False \
--dataset=tiny_imagenet \
--setting=top1 \
--cap_model=v_base \
--depth=14 \
--max_epoch=500 \
--schedule 200 300 400 \
--batch_size_train=512 \
--batch_size_test=256 \
--lr=0.01 \
--optim=sgd \
--s35


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
