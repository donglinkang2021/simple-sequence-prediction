# python train.py --multirun \
#     model=lstm_linear1,lstm_linear2 \
#     model.hidden_size=32,64,128,256,512 \
#     train.time_steps=4,8,12,16,32,48,64,80,96,112,128 \
#     train.learning_rate=1e-3,1e-4,1e-5,3e-3,3e-4,3e-5,5e-3,5e-4,5e-5,7e-3,7e-4,7e-5

# last time min mse is 0.0138
python train.py --multirun \
    model=lstm_linear1,lstm_linear2 \
    model.hidden_size=64,512 \
    model.is_rand_init=False,True \
    train.learning_rate=1e-3,5e-4 \
    train.time_steps=112,128 \
    train.batch_size=8,16,32,64

