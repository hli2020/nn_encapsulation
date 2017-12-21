#!/bin/bash

start=`date +%s`

# based on 'cifar_base_104_no_relu_multi_crop.sh'
# train and test
CUDA_VISIBLE_DEVICES=1 python holly_cifar.py \
--experiment_name=cifar_base_104_no_relu_droput_c \
--dataset=cifar \
--model_cifar=capsule \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.01 \
--route_num=4 \
--add_cap_dropout \
--dropout_p=0.1 \
--multi_crop_test \
--deploy

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
