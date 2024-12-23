#!/bin/bash

model_names=(
mhvqlnmlp_rope_vs512_dm64_nh8_y_w_ts96_lr0.007/2024-12-23-00-04-22 
mhvqlnmlp_vs512_dm64_nh2_y_w_ts96_lr0.007/2024-12-23-00-04-12      
mhvqlnpemlp_vs512_dm64_nh2_y_w_ts96_lr0.007/2024-12-23-00-04-16    
vqlnmlp_rope_vs512_dm64_y_w_ts96_lr0.007/2024-12-23-00-05-26       
vqlnmlp_vs512_dm64_y_w_ts96_lr0.007/2024-12-23-00-05-16            
vqlnpemlp_vs512_dm64_y_w_ts96_lr0.007/2024-12-23-00-05-20          
vqlnmlp_sinpe_vs512_dm64_y_w_ts96_lr0.007/2024-12-23-00-05-23      
mhvqlnmlp_sinpe_vs512_dm64_nh8_y_w_ts96_lr0.007/2024-12-23-00-04-19
)

log_dir=./logs

for model_name in "${model_names[@]}"; do
    python predict.py \
        --model_name "$model_name" \
        --log_dir "$log_dir" \
        --predict_rate 1 \
        --save_img_dir predict_images
done

# bash scripts/predict.sh