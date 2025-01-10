"""
TODO: 
1. implement LSTM model from scratch, refer to https://github.com/donglinkang2021/makemore/blob/master/model/rnn.py
2. implement RNN or GRU model from scratch
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int, output_dim:int):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x:torch.Tensor):
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :])

if __name__ == "__main__":
    from .utils import init_weights, model_summary
    import numpy as np
    B, T, D = 32, 10, 1
    model = Model(input_dim=D, hidden_dim=64, num_layers=2, output_dim=1)
    model.apply(init_weights)
    model_summary(model)

    seq = np.random.randn(B, T, D).astype(np.float32)
    seq = torch.from_numpy(seq)
    out = model(seq)
    print(out.shape)  # Expected output: torch.Size([32, 1])

# python -m src.model.lstm
