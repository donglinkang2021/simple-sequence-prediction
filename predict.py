import torch
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import argparse

def predict_regressive_1d(model:torch.nn.Module, seq: np.ndarray, n_steps: int, device: torch.device) -> np.ndarray:
    model.eval()
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) # (1, T, 1)
    seq = seq.to(device)
    predictions = []
    with torch.no_grad():
        for _ in tqdm(range(n_steps)):
            out = model(seq)[0]
            if out.shape[1] != 1:
                out = out[:, -1, :]
            predictions.append(out.squeeze().cpu().numpy())
            seq = torch.cat((seq[:, 1:, :], out.unsqueeze(-1)), dim=1)
    return np.array(predictions)

def predict_batch(model:torch.nn.Module, seq: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Input seq: numpy array of shape (n_steps, time_steps, D) or (n_steps, time_steps)
    Return predictions: numpy array of shape (n_steps,)
    """
    model.eval()
    seq = torch.tensor(seq, dtype=torch.float32) 
    # make it (n_steps, T, D)
    if len(seq.shape) == 2:
        seq = seq.unsqueeze(-1)
    seq = seq.to(device)
    output = model(seq)[0]
    if output.shape[1] != 1:
        output = output[:, -1, :]
    return output.detach().cpu().squeeze().numpy()

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def main():
    parser = argparse.ArgumentParser(description='Predict using a trained LSTM model.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for prediction.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory where the model logs are stored.')
    parser.add_argument('--predict_rate', type=float, default=1, help='Rate at which to predict.')
    parser.add_argument('--save_img_dir', type=str, default='predict_images', help='Directory to save prediction images.')
    parser.add_argument('--predict_on_testset', action='store_true', help='Flag to predict on the training set.')

    args = parser.parse_args()

    model_name = args.model_name
    log_dir = args.log_dir
    predict_rate = args.predict_rate
    save_img_dir = args.save_img_dir
    predict_on_testset = args.predict_on_testset

    # load model and mkdir
    Path(save_img_dir).mkdir(parents=True, exist_ok=True)
    model_prefix = model_name.split('/')[0]
    model_prefix = f'{save_img_dir}/{model_prefix}'
    model_dir = f'{log_dir}/{model_name}'
    cfg = OmegaConf.load(f'{model_dir}/config.yaml')
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(f'{log_dir}/{model_name}/model.pth', weights_only=True))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    from src.utils.data1d import load_data, prepare_sequences
    file2short = {
        'data/w+y.csv': 'w_y',
        'data/y+w.csv': 'y_w',
    }
    test_file = cfg.dataset.train_file
    if predict_on_testset:
        test_file = "data/y+w.csv" if test_file == "data/w+y.csv" else "data/w+y.csv"

    # predict one-step and multi-step
    seq = load_data(test_file)
    mean = seq.mean()
    std = seq.std()
    seq_norm = (seq - mean) / std
    time_steps = cfg.train.time_steps
    sequences, _ = prepare_sequences(seq_norm, time_steps)
    total_length = len(sequences)
    n_steps = int(total_length * predict_rate)
    test_seq = sequences[-n_steps:]
    predictions_batch = predict_batch(model, test_seq, device)
    predictions_batch = predictions_batch * std + mean
    predictions_regressive = predict_regressive_1d(model, seq_norm[:-n_steps], n_steps, device)
    predictions_regressive = predictions_regressive * std + mean

    # plot results
    from src.utils.plot import plot_results
    import pandas as pd
    data = pd.read_csv(test_file).to_numpy()
    plot_results(predictions_batch, data, save_path=f'{model_prefix}_predict_batch_{file2short[test_file]}.png')
    print(f'MSE loss: {mse_loss(data[:,1][-n_steps:], predictions_batch)}')
    plot_results(predictions_regressive, data, save_path=f'{model_prefix}_predict_regressive_{file2short[test_file]}.png')
    print(f'MSE loss: {mse_loss(data[:,1][-n_steps:], predictions_regressive)}')

if __name__ == '__main__':
    main()
