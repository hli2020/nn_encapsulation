#!/bin/bash

start=`date +%s`

# train
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
--save_folder=renew_512_new_scale --deploy --batch_size=32 --ssd_dim=512 --max_iter=80100

# test around 8w
CUDA_VISIBLE_DEVICES=7 python eval.py --experiment_name=renew_512_new_scale \
--trained_model=ssd512_0712_iter_80100.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

# train
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
--save_folder=renew_512_new_scale --deploy --batch_size=32 \
--ssd_dim=512 --max_iter=100100 --resume=ssd512_0712_iter_80100

# test around 10w
CUDA_VISIBLE_DEVICES=7 python eval.py --experiment_name=renew_512_new_scale \
--trained_model=ssd512_0712_iter_100100.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

# train
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
--save_folder=renew_512_new_scale --deploy --batch_size=32 \
--ssd_dim=512 --max_iter=120100 --resume=ssd512_0712_iter_100100

# test around 12w
CUDA_VISIBLE_DEVICES=7 python eval.py --experiment_name=renew_512_new_scale \
--trained_model=ssd512_0712_iter_120100.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

# train
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
--save_folder=renew_512_new_scale --deploy --batch_size=32 \
--ssd_dim=512 --resume=ssd512_0712_iter_120100

# test around 12w
CUDA_VISIBLE_DEVICES=7 python eval.py --experiment_name=renew_512_new_scale \
--trained_model=final_v2.pth \
--ssd_dim=512 --conf_thresh=0.005 --top_k=300 --nms_thresh=0.45

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
