import torch
import torch.nn as nn
import math
from einops import rearrange
from simplepe import SinusoidalPositionalEncoding

class Model(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, input_dim:int, output_dim:int, block_size:int, n_head:int):
        super(Model, self).__init__()
        assert d_model % n_head == 0, 'd_model should be divisible by n_head'
        self.d_model = d_model
        self.n_head = n_head
        # register buffer here for quantization
        head_size = d_model // n_head
        self.register_buffer('embedding', torch.randn(n_head, vocab_size, head_size, requires_grad=True))
        self.pe = SinusoidalPositionalEncoding(head_size, max_len=block_size)
        self.encoder = nn.Linear(input_dim, d_model)
        self.ln = nn.LayerNorm(head_size)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # input x shape: (B, T, 1)
        x = self.encoder(x) # -> (B, T, D)
        x = rearrange(x, 'b t (nh hs) -> b nh t hs', nh=self.n_head)
        x = self.pe(x)
        x = self.ln(x)
        probs = (x @ self.embedding.transpose(-2,-1) / math.sqrt(self.d_model)).softmax(dim=-1)
        quantize = probs @ self.embedding
        quantize = rearrange(quantize, 'b nh t hs -> b t (nh hs)')
        x = self.decoder(quantize)
        return x, None

if __name__ == "__main__":
    from .utils import init_weights, model_summary
    import numpy as np
    B, T, D = 32, 10, 1
    model = Model(vocab_size=32, d_model=64, input_dim=D, output_dim=1, block_size=512, n_head=2)
    model.apply(init_weights)
    model_summary(model)

    seq = np.random.randn(B, T, D).astype(np.float32)
    seq = torch.from_numpy(seq)
    out = model(seq)[0]
    print(out.shape)  # Expected output: torch.Size([32, 10, 1])

# python -m src.model.mhvqlnmlp_sinpe


        
