#!/bin/bash

start=`date +%s`

# train and test
CUDA_VISIBLE_DEVICES=6 python holly_cifar.py \
--experiment_name=capsule_202_v4_4 \
--dataset=cifar10 \
--model_cifar=capsule \
--cap_model=v4_4 \
--cap_N=3 \
--epochs=300 \
--schedule_cifar 150 225 \
--optim=rmsprop \
--lr=0.0001 \
--route_num=2 \
--multi_crop_test \
--port=2000

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
