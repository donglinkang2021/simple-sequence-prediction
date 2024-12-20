import torch
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

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
    model_name = 'vqmlp_vs32_dm8_y_w_ts96_lr0.007/2024-12-20-11-14-50'
    # model_name = 'lstm_linear1_y_w_ts112_lr0.0005/2024-12-20-10-47-35'
    log_dir = '.cache/logs-vqmlp-vs-dm-lr-ts'
    # log_dir = './logs'
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
    print(f"MSE loss: {mse_loss(seq_norm[-n_steps:], predictions_batch)}")
    predictions_batch = predictions_batch * std + mean

    from src.utils.plot import plot_results
    import pandas as pd
    data = pd.read_csv(test_file).to_numpy()

    model_prefix = model_name.split('_')[0]
    plot_results(predictions_batch, data, save_path=f'{model_prefix}_predict_batch_y_w.png')
    print(f'MSE loss: {mse_loss(data[:,1][-n_steps:], predictions_batch)}')
    predictions_regressive = predict_regressive_1d(model, seq_norm[:-n_steps], n_steps, device)
    predictions_regressive = predictions_regressive * std + mean
    plot_results(predictions_regressive, data, save_path=f'{model_prefix}_predict_regressive_y_w.png')

if __name__ == '__main__':
    main()
