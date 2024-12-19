python train.py --multirun \
    model=lstm_linear1 \
    train.time_steps=4,8,12,16,32,48,64,80,96,112,128 \
    train.learning_rate=1e-3,1e-4,1e-5,3e-3,3e-4,3e-5,5e-3,5e-4,5e-5,7e-3,7e-4,7e-5
