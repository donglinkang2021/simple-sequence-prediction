#!/bin/bash

model_names=(
lstm_y_w_ts112_lr0.0005/2025-01-10-21-50-56
att_y_w_ts112_lr0.003/2025-01-10-23-06-46
att_mh_y_w_ts112_lr0.0003/2025-01-10-23-15-54
vq_y_w_ts112_lr0.001/2025-01-10-23-17-38
vq_mh_y_w_ts112_lr0.003/2025-01-11-00-01-54
)

log_dir=./logs
# log_dir=./.cache

for model_name in "${model_names[@]}"; do
    python predict.py \
        --model_name "$model_name" \
        --log_dir "$log_dir" \
        --predict_rate 1 \
        --save_img_dir predict_images
done

# bash scripts/predict.sh