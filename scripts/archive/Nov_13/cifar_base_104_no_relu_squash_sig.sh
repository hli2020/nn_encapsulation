#!/bin/bash

start=`date +%s`

# train and test
# '--test_batch=32' has to be small on cifar100
CUDA_VISIBLE_DEVICES=7 python holly_cifar.py \
--experiment_name=cifar_base_104_no_relu_squash_sig \
--dataset=cifar10 \
--model_cifar=capsule \
--cap_model=v0 \
--epochs=300 \
--schedule_cifar 150 225 \
--lr=0.0001 \
--optim=sgd \
--route_num=2 \
--w_version=v2 \
--squash_manner=sigmoid \
--multi_crop_test \
--port=2000

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
