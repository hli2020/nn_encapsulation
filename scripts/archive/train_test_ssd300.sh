#!/bin/bash

start=`date +%s`

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --save_folder=renew_300_new_scale --batch_size=32 --ssd_dim=300 --deploy

# best setting compared with the following two
#CUDA_VISIBLE_DEVICES=3 python test.py --experiment_name=renew_300_new_scale \
#--trained_model=final_v2.pth --ssd_dim=300 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

#CUDA_VISIBLE_DEVICES=3 python test.py --experiment_name=renew_300_new_scale \
#--trained_model=final_v2.pth --sub_folder_suffix=set1 --ssd_dim=300 --conf_thresh=0.005 --top_k=2000 --nms_thresh=0.5

#CUDA_VISIBLE_DEVICES=3 python test.py --experiment_name=renew_300_new_scale \
#--trained_model=final_v2.pth --sub_folder_suffix=set2 --ssd_dim=300 --conf_thresh=0.01 --top_k=300 --nms_thresh=0.5

# other iterations
CUDA_VISIBLE_DEVICES=3 python eval.py --experiment_name=renew_300_new_scale \
--trained_model=ssd300_0712_iter_125000.pth --ssd_dim=300 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

CUDA_VISIBLE_DEVICES=3 python eval.py --experiment_name=renew_300_new_scale \
--trained_model=ssd300_0712_iter_105000.pth --ssd_dim=300 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

CUDA_VISIBLE_DEVICES=3 python eval.py --experiment_name=renew_300_new_scale \
--trained_model=ssd300_0712_iter_85000.pth --ssd_dim=300 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
