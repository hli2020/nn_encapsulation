#!/bin/bash

start=`date +%s`

# train
#CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
#--experiment_name=ssd_base_101 --deploy \
#--dataset=coco \
#--batch_size=32 --ssd_dim=512 --max_iter=200000 \
#--prior_config=v2_512 --lr=1e-3 --schedule=100000,140000,160000 --gamma=0.5

## test
# '0712' is shown for legacy reason
CUDA_VISIBLE_DEVICES=2 python test.py --experiment_name=ssd_base_101 \
--trained_model=ssd512_0712_iter_10000.pth \
--dataset=coco \
--show_freq=20 \
--sub_folder_suffix=set2 \
--ssd_dim=512 --conf_thresh=0.01 --top_k=300 --nms_thresh=0.5 \
--prior_config=v2_512


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
