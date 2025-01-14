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
            out = model(seq)
            if out.shape[1] != 1:
                out = out[:, -1, :]
            predictions.append(out.squeeze().cpu().numpy())
            seq = torch.cat((seq[:, 1:, :], out.unsqueeze(-1)), dim=1)
    return np.array(predictions)

def predict_batch(model:torch.nn.Module, seq: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    """
    Input seq: numpy array of shape (n_steps, time_steps, D) or (n_steps, time_steps)
    Return predictions: numpy array of shape (n_steps,)
    """
    model.eval()
    predictions = []
    
    # Process sequence in batches
    for i in range(0, len(seq), batch_size):
        batch = seq[i:i + batch_size]
        batch = torch.tensor(batch, dtype=torch.float32)
        
        # make it (batch_size, T, D)
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(-1)
        
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch)
            if output.shape[1] != 1:
                output = output[:, -1, :]
            predictions.append(output.cpu().squeeze().numpy())
    
    return np.concatenate(predictions)

def mse_loss(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def predict(model_name:str, log_dir:str, predict_rate:float, save_img_dir:str):
    # load model and mkdir
    pred_prefix = f'{save_img_dir}/{model_name}'
    Path(pred_prefix).mkdir(parents=True, exist_ok=True)
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
        'data/4single.csv': '4single'
    }
    test_file = cfg.dataset.train_file
    # test_file = "data/4single.csv"
    # test_file = "data/w+y.csv"

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
    plot_results(predictions_batch, data, save_path=f'{pred_prefix}/predict_batch_{file2short[test_file]}.png')
    plot_results(predictions_regressive, data, save_path=f'{pred_prefix}/predict_regressive_{file2short[test_file]}.png')
    metrics = {
        'Predict/mse_batch': mse_loss(data[:,1][-n_steps:], predictions_batch),
        'Predict/mse_regressive': mse_loss(data[:,1][-n_steps:], predictions_regressive)
    }
    OmegaConf.save(OmegaConf.create(metrics), f'{pred_prefix}/metrics_{file2short[test_file]}.yaml')
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Predict using a trained LSTM model.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for prediction.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory where the model logs are stored.')
    parser.add_argument('--predict_rate', type=float, default=1, help='Rate at which to predict.')
    parser.add_argument('--save_img_dir', type=str, default='predict_images', help='Directory to save prediction images.')
    args = parser.parse_args()
    predict(**vars(args))

if __name__ == '__main__':
    main()
