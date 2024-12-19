import torch
import torch.nn as nn

class BaseLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int, is_rand_init:bool=False):
        super(BaseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_rand_init = is_rand_init

    def forward(self, x, hidden=None):
        raise NotImplementedError
    
    def init_hidden(self, batch_size:int, device:torch.device):
        if self.is_rand_init:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)
