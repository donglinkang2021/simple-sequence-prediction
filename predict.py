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

def predict_batch(model:BaseLSTM, seq: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Input seq: numpy array of shape (n_steps, time_steps, 2)
    Return predictions: numpy array of shape (n_steps,)
    """
    model.eval()
    seq = torch.tensor(seq, dtype=torch.float32) # (n_steps, T, D)
    seq = seq.to(device)
    return model.forward(seq)[0].detach().cpu().squeeze().numpy()


def main():
    model_name = 'lstm_linear1_y_w_ts128_lr0.007/2024-12-19-13-38-36'
    model_dir = f'logs/{model_name}'
    cfg = OmegaConf.load(f'{model_dir}/config.yaml')
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(f'logs/{model_name}/model.pth', weights_only=True))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    from src.utils.data import load_data, prepare_sequences
    
    data = load_data(cfg.dataset.train_file)

    seq = data.to_numpy()

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
    plot_results(predictions_batch, seq, save_path=f'predict_batch.png')

if __name__ == '__main__':
    main()
