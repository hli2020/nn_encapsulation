#!/bin/bash

start=`date +%s`

# train
# about bs: 2gpu, bs=16, out of mem; bs=10 is ok
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
--save_folder=renew_634 --batch_size=32 --ssd_dim=634 --max_iter=100000 \
--prior_config=v2_634 --lr=1e-3 --schedule=60000,80000,90000 --gamma=0.5 \
--deploy

# test
CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_634 \
--trained_model=ssd634_0712_iter_65000.pth \
--ssd_dim=634 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_634

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_634 \
--trained_model=ssd634_0712_iter_75000.pth \
--ssd_dim=634 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_634

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_634 \
--trained_model=ssd634_0712_iter_85000.pth \
--ssd_dim=634 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_634

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_634 \
--trained_model=final_v2.pth \
--ssd_dim=634 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_634


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
