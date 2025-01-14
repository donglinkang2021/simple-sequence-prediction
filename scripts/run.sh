# baseline model
# python train.py --multirun \
#     model=lstm \
#     dataset=4single \
#     train.learning_rate=3e-3,1e-3,5e-4,3e-4,1e-4 \
#     train.time_steps=64,128,256,512,1024

# our model
python train.py --multirun \
    train.is_align_target=True \
    model=att,vq,att_mh,vq_mh \
    dataset=4single \
    ++model.block_size=2048 \
    train.learning_rate=3e-3,1e-3,5e-4,3e-4,1e-4 \
    train.time_steps=64,128,256,512,1024
