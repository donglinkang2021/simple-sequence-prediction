# python train.py \
#     train.is_align_target=True \
#     model=att \
#     dataset=y_w \
#     train.learning_rate=5e-4 \
#     train.time_steps=112 \
#     train.batch_size=64

CUDA_VISIBLE_DEVICES=0 python train.py --multirun \
    train.is_align_target=True \
    model=att \
    dataset=y_w \
    model.is_inln=true,false \
    model.is_qln=true,false \
    model.is_kln=true,false \
    model.is_n_pe=true,false \
    model.is_causal=true,false \
    model.pe_type=none,randpe,sinpe,rope \
    train.learning_rate=3e-3,1e-3,5e-4 \
    train.time_steps=112 \
    train.batch_size=64
