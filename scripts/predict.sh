#!/bin/bash

model_names=(
lstm_y_w_ts112_lr0.0005/2025-01-10-21-08-32
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