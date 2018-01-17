#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=4 python holly_capsule.py \
--experiment_name=base_101_v10_2 \
--debug_mode=False \
--dataset=cifar10 \
--cap_model=v0 \
--max_epoch=300 \
--loss_form=margin \
--optim=adam \
--schedule 150 200 250 \
--lr=0.0001 \
--w_version=v3 \
--fc_time=1


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
