#!/bin/bash

start=`date +%s`

# train
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
--save_folder=renew_512_set2 --deploy --batch_size=32 --ssd_dim=512 --max_iter=100000 \
--prior_config=v2_512 --lr=1e-3 --schedule=60000,80000,90000 --gamma=0.5

# test
CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set2 \
--trained_model=final_v2.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_512


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
