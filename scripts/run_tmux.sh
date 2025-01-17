# now we trained auto-regressive best model
# python train.py --multirun \
#     model=vqlnpemlp train.is_align_target=True \
#     model.vocab_size=4,8,16,32,64,128,256,512 \
#     model.d_model=4,8,16,32,64,128,256,512 \
#     train.time_steps=4,8,12,16,32,48,64,80,96,112,128 \
#     train.learning_rate=1e-3,1e-4,1e-5,3e-3,3e-4,3e-5,5e-3,5e-4,5e-5,7e-3,7e-4,7e-5

#!/bin/bash

vocab_sizes=(16 32 64 128 256 512 768 1024)
session="multi_run_mhvq"

tmux new-session -d -s $session

for i in ${!vocab_sizes[@]}; do
    if [ $i -ne 0 ]; then
        tmux new-window -t $session
    fi
    vocab=${vocab_sizes[$i]}
    tmux send-keys -t $session:$i "conda activate linkdom" C-m
    tmux send-keys -t $session:$i "export CUDA_VISIBLE_DEVICES=$i" C-m
    tmux send-keys -t $session:$i "python train.py --multirun model=mhvqlnpemlp,mhvqlnmlp train.is_align_target=True model.d_model=32,64,128,256,512 model.n_head=2,4,8,16 train.time_steps=96 train.learning_rate=3e-2,1e-2,7e-3,3e-3,7e-4,3e-4 ++model.vocab_size=$vocab" C-m
done

tmux attach -t $session