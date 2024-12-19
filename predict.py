import torch
import numpy as np
from src.model.base import BaseLSTM
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

def predict_regressive(model:BaseLSTM, seq: np.ndarray, n_steps: int, device: torch.device) -> np.ndarray:
    """
    Input seq: numpy array of shape (T - n_steps, 2), we should norm it before input
    Return predictions: numpy array of shape (n_steps,)
    >>> predictions_regressive = predict_regressive(model, seq_norm[:-n_steps], n_steps, device)
    >>> predictions_regressive = predictions_regressive * std + mean
    >>> plot_results(predictions_regressive, seq, save_path=f'predict_regressive.png')
    """
    model.eval()
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0) # (1, T, D)
    seq = seq.to(device)
    seq_diff = (seq[0, 1:, 0] - seq[0, :-1, 0]).mean()
    predictions = []
    hidden = model.init_hidden(seq.size(0), device)
    with torch.no_grad():
        for _ in tqdm(range(n_steps)):
            out, hidden = model.forward(seq, hidden)
            predictions.append(out.squeeze().cpu().numpy())
            seq_1 = torch.cat((seq[:, 1:, 1].unsqueeze(-1), out.unsqueeze(0)), dim=1)
            seq_0 = seq[:, :, 0].unsqueeze(-1) + seq_diff
            seq = torch.cat((seq_0, seq_1), dim=-1)
    return np.array(predictions)

def predict_regressive_1d(model:BaseLSTM, seq: np.ndarray, n_steps: int, device: torch.device) -> np.ndarray:
    model.eval()
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) # (1, T, 1)
    seq = seq.to(device)
    predictions = []
    hidden = model.init_hidden(seq.size(0), device)
    with torch.no_grad():
        for _ in tqdm(range(n_steps)):
            out, hidden = model.forward(seq, hidden)
            predictions.append(out.squeeze().cpu().numpy())
            seq = torch.cat((seq[:, 1:, :], out.unsqueeze(-1)), dim=1)
    return np.array(predictions)

def predict_batch(model:BaseLSTM, seq: np.ndarray, device: torch.device) -> np.ndarray:
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
    return model.forward(seq)[0].detach().cpu().squeeze().numpy()


def main():
    model_name = 'lstm_linear1_hs64_y_w_ts112_lr0.0005/2024-12-19-16-25-27'
    log_dir = '.cache/logs-model-hs-isr-lr-ts-bs'
    model_dir = f'{log_dir}/{model_name}'
    cfg = OmegaConf.load(f'{model_dir}/config.yaml')
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(f'{log_dir}/{model_name}/model.pth', weights_only=True))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    from src.utils.data1d import load_data, prepare_sequences
    
    test_file = cfg.dataset.train_file
    # test_file = "data/w+y.csv"
    seq = load_data(test_file)

    predict_rate = 1
    
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

    from src.utils.plot import plot_results
    import pandas as pd
    data = pd.read_csv(test_file).to_numpy()
    # plot_results(predictions_batch, data, save_path='predict_batch_w_y.png')

    predictions_regressive = predict_regressive_1d(model, seq_norm[:-n_steps], n_steps, device)
    predictions_regressive = predictions_regressive * std + mean
    plot_results(predictions_regressive, data, save_path=f'predict_regressive_y_w.png')

if __name__ == '__main__':
    main()
