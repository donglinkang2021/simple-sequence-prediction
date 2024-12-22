#!/bin/bash

model_names=(
    mhvqlnpemlp_vs768_dm128_nh16_y_w_ts96_lr0.007/2024-12-21-17-08-13
    mhvqlnpemlp_vs128_dm64_nh16_y_w_ts96_lr0.01/2024-12-21-17-00-40  
    mhvqlnpemlp_vs128_dm512_nh16_y_w_ts96_lr0.03/2024-12-21-17-09-08 
    mhvqlnpemlp_vs1024_dm128_nh16_y_w_ts96_lr0.01/2024-12-21-17-09-52
    mhvqlnpemlp_vs768_dm64_nh16_y_w_ts96_lr0.007/2024-12-21-17-02-01 
)

log_dir=./logs

for model_name in "${model_names[@]}"; do
    python predict.py \
        --model_name "$model_name" \
        --log_dir "$log_dir" \
        --predict_rate 1 \
        --save_img_dir predict_images
done

# bash predict.sh