# CUDA_VISIBLE_DEVICES=3 python train.py \
#     train.is_align_target=True \
#     model=vq_mh \
#     dataset=y_w \
#     train.learning_rate=5e-4 \
#     train.time_steps=112 \
#     train.batch_size=64

CUDA_VISIBLE_DEVICES=3 python train.py --multirun \
    train.is_align_target=True \
    model=vq_mh \
    dataset=y_w \
    model.vocab_size=32,64,128,256\
    model.kv_heads=1,2,4,8 \
    model.is_inln=true,false \
    model.is_qln=true,false \
    model.is_causal=true,false \
    model.is_n_pe=true,false \
    model.pe_type=sinpe,rope \
    train.learning_rate=3e-3,1e-3,5e-4 \
    train.time_steps=112 \
    train.batch_size=64
