#!/bin/bash

start=`date +%s`

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set2 \
--trained_model=ssd512_0712_iter_60000.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_512

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set2 \
--trained_model=ssd512_0712_iter_55000.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_512

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set2 \
--trained_model=ssd512_0712_iter_75000.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_512

CUDA_VISIBLE_DEVICES=1 python eval.py --experiment_name=renew_512_set2 \
--trained_model=ssd512_0712_iter_80000.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45 \
--prior_config=v2_512


end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
