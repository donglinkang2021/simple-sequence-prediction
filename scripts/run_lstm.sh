# last time 
# best model
# min mse is 0.0137 on lstm
python train.py \
    model=lstm \
    dataset=y_w \
    model.hidden_dim=64 \
    train.learning_rate=5e-4 \
    train.time_steps=112 \
    train.batch_size=64
