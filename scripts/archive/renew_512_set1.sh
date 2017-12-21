#!/bin/bash

start=`date +%s`

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--save_folder=renew_512_set1 --deploy --batch_size=32 --ssd_dim=512 --max_iter=82000 \
--prior_config=v2_512

# test
CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set1 \
--trained_model=final_v2.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_512


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
