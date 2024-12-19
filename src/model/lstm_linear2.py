import torch
import torch.nn as nn
import numpy as np

from .base import BaseLSTM

class Model(BaseLSTM):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int, is_rand_init:bool=False):
        super(Model, self).__init__(input_size, hidden_size, num_layers, output_size, is_rand_init)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, output_size)
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.head(lstm_out[:, -1, :])
        return out, hidden

if __name__ == "__main__":
    B, T, D = 32, 10, 2

    from .utils import init_weights, model_summary
    model = Model(input_size=D, hidden_size=64, num_layers=2, output_size=1)
    model.apply(init_weights)
    model_summary(model)

    seq = np.random.randn(B, T, D).astype(np.float32)
    seq = torch.from_numpy(seq)
    hidden = model.init_hidden(B, torch.device('cpu'))
    out, hidden = model(seq, hidden)
    print(out.shape)  # Expected output: torch.Size([32, 1])
    print(hidden[0].shape)  # Expected output: torch.Size([32, 1])
    print(hidden[1].shape)  # Expected output: torch.Size([32, 1])

# python -m src.model.lstm_linear2
