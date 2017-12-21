#!/bin/bash

start=`date +%s`

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --save_folder=renew_300_set1 \
--batch_size=32 --ssd_dim=300 --deploy --prior_config=v3  --max_iter=82000

CUDA_VISIBLE_DEVICES=3 python eval.py --experiment_name=renew_300_set1 \
--trained_model=final_v2.pth --ssd_dim=300 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
