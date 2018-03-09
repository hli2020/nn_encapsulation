#!/bin/bash

start=`date +%s`

#--experiment_name=base_101_v4_rerun \
# train and test
CUDA_VISIBLE_DEVICES=1,2,6,7 python holly_capsule.py \
--experiment_name=base_201_v2_rerun_FUCK_3 \
--debug_mode=False \
--dataset=tiny_imagenet \
--setting=top1 \
--cap_model=v0 \
--num_workers=8 \
--route_num=3 \
--max_epoch=50000000000 \
--loss_form=margin \
--optim=adam \
--schedule 200 300 400 \
--batch_size_train=100 \
--batch_size_test=100 \
--primary_cap_num=32 \
--pre_ch_num=256 \
--lr=0.0001 \
--bigger_input \
--s35

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
