# python train.py --multirun \
#     model=lstm_linear1,lstm_linear2 \
#     model.hidden_size=32,64,128,256,512 \
#     train.time_steps=4,8,12,16,32,48,64,80,96,112,128 \
#     train.learning_rate=1e-3,1e-4,1e-5,3e-3,3e-4,3e-5,5e-3,5e-4,5e-5,7e-3,7e-4,7e-5

# last time 
# best model
# min mse is 0.0137 on lstm_linear1
# python train.py \
#     model=lstm_linear1 \
#     model.hidden_size=64 \
#     train.learning_rate=5e-4 \
#     train.time_steps=112 \
#     train.batch_size=64

# # best vqmlp 0.0261
# python train.py \
#     model=vqmlp train.is_align_target=True \
#     model.vocab_size=32 \
#     model.d_model=8 \
#     train.time_steps=96 \
#     train.learning_rate=7e-3

# python train.py \
#     model=vqlnmlp train.is_align_target=True \
#     model.vocab_size=32 \
#     model.d_model=8 \
#     train.time_steps=96 \
#     train.learning_rate=7e-3

python train.py --multirun \
    model=vqlnpemlp train.is_align_target=True \
    model.vocab_size=32 \
    model.d_model=8 \
    train.time_steps=96 \
    train.learning_rate=3e-2,1e-2,7e-3,3e-3
