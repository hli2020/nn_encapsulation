#!/bin/bash

start=`date +%s`

# train
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#--experiment_name=ssd_base_102 --deploy \
#--dataset=coco \
#--batch_size=32 --ssd_dim=512 --max_iter=80000 \
#--prior_config=v2_512 --lr=0.001 --weight_decay=1e-3 --schedule=40000,60000,700000 --gamma=0.1

## test
# '0712' is shown for legacy reason
CUDA_VISIBLE_DEVICES=2 python test.py --experiment_name=ssd_base_102 \
--trained_model=ssd512_COCO_iter_40000.pth \
--dataset=coco \
--show_freq=20 \
--sub_folder_suffix=set1 \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.5 \
--prior_config=v2_512


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
