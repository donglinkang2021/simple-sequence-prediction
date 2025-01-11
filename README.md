

<div align="center">

# Simple Sequence Prediction

<img src="https://img.shields.io/badge/license-MIT-8bb903" alt="License">
<img src="https://img.shields.io/badge/python-3.9.20-3776AB" alt="Python">
<img src="https://img.shields.io/badge/pytorch-2.5.1+cu124-EE4C2C" alt="PyTorch">
<img src="https://img.shields.io/badge/hydra-1.3.2-6ca6c0" alt="Hydra">
<img src="https://img.shields.io/badge/tensorboard-2.18.0-FF6F00" alt="TensorBoard">

</div>

<div align=center>
   <img src="docs/images/20250111/att_mh_y_w_ts112_lr0.0003/2025-01-10-23-15-54/predict_regressive_y_w.png" width="50%">
</div>

This project implements several sequence prediction models with PyTorch. The task is to predict formaldehyde concentration based on time.


### Setup

1. Clone the repository:
   
   ```bash
   git clone https://github.com/donglinkang2021/simple-sequence-prediction.git
   cd simple-sequence-prediction
   ```

2. Install the required dependencies:
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Usage

To train the model, run the following command:

```bash
# recommend to remove all logs before running
# rm -rf logs # remove all logs
# also recommend use tmux to run the following scripts
bash scripts/run_lstm.sh # simple baseline
bash scripts/run_att.sh # attention mechanism
bash scripts/run_att_mh.sh # multi-head attention mechanism
bash scripts/run_vq.sh # vector quantization mlp
bash scripts/run_vq_mh.sh # vector quantization multi-head mlp
tensorboard --logdir=logs --bind_all # start tensorboard
python scripts/search_tb_event.py # search the best result from tensorboard event files
```

If you want to train on multi-windows using `tmux`(multi GPU) once, you can refer to the following command:

```bash
bash scripts/run_tmux.sh
```

You can copy the best model name to `scripts/predict.sh`, run the following command:

```bash
bash scripts/predict.sh
# or just run the following command
python predict.py --model_name att_mh_y_w_ts112_lr0.0003/2025-01-10-23-15-54
```

### License

This project is licensed under the MIT License.
