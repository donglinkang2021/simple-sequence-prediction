# model_name=lstm_linear1_y_w_ts112_lr0.0005/2024-12-20-22-23-02
# model_name=vqlnmlp_y_w_ts96_lr0.007/2024-12-20-22-23-13
model_name=vqlnpemlp_vs512_dm32_y_w_ts96_lr0.007/2024-12-20-22-55-08
log_dir=./logs
python predict.py \
    --model_name $model_name \
    --log_dir $log_dir \
    --predict_rate 1 \
    --save_img_dir predict_images

# bash predict.sh