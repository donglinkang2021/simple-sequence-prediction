import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, input_dim:int, output_dim:int, block_size:int):
        super(Model, self).__init__()
        self.d_model = d_model
        # register buffer here for quantization
        self.register_buffer('embedding', torch.randn(vocab_size, d_model, requires_grad=True))
        self.register_buffer('wpe', 0.01 * torch.randn(block_size, d_model))
        self.encoder = nn.Linear(input_dim, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # input x shape: (B, T, 1)
        pos = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        pos_emb = self.wpe[pos]
        x = self.encoder(x) # -> (B, T, D)
        x = x + pos_emb
        x = self.ln(x)
        probs = (x @ self.embedding.T / math.sqrt(self.d_model)).softmax(dim=-1)
        quantize = probs @ self.embedding
        x = self.decoder(quantize)
        return x, probs.argmax(dim=-1)

if __name__ == "__main__":
    from .utils import init_weights, model_summary
    import numpy as np
    B, T, D = 32, 10, 1
    model = Model(vocab_size=16, d_model=64, input_dim=D, output_dim=1, block_size=512)
    model.apply(init_weights)
    model_summary(model)

    seq = np.random.randn(B, T, D).astype(np.float32)
    seq = torch.from_numpy(seq)
    out = model(seq)[0]
    print(out.shape)  # Expected output: torch.Size([32, 10, 1])

# python -m src.model.vqlnpemlp


        
