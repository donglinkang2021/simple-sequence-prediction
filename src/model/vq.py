import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_pe

class Model(nn.Module):
    def __init__(
            self, 
            vocab_size:int, 
            d_model:int, 
            input_dim:int, 
            output_dim:int, 
            block_size:int, 
            is_inln: bool = False,
            is_causal: bool = False,
            pe_type: str = None,  # 'randn', 'sinpe', 'rope', None
        ):
        super(Model, self).__init__()

        self.is_inln = is_inln
        self.is_causal = is_causal

        self.fc_in = nn.Linear(input_dim, d_model)
        
        if is_inln:
            self.ln_in = nn.LayerNorm(d_model)

        self.pe = get_pe(pe_type, d_model, block_size)

        self.cache = nn.Parameter(
            torch.randn(vocab_size, d_model), requires_grad=True
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # input x shape: (B, T, input_dim)
        x = self.fc_in(x)

        if self.pe is not None:
            x = self.pe(x)

        if self.is_inln:
            x = self.ln_in(x)

        xo = F.scaled_dot_product_attention(x, self.cache, self.cache, is_causal=self.is_causal)
        return self.fc_out(xo)

if __name__ == "__main__":
    from .utils import init_weights, model_summary
    import numpy as np
    B, T, D = 32, 10, 1
    model = Model(
        vocab_size=32, 
        d_model=64, 
        input_dim=D, 
        output_dim=1, 
        block_size=218, 
        is_inln=True,
        is_causal=True,
        pe_type='rope'
    )
    model.apply(init_weights)
    model_summary(model)

    seq = np.random.randn(B, T, D).astype(np.float32)
    seq = torch.from_numpy(seq)
    out = model(seq)
    print(out.shape)  # Expected output: torch.Size([32, 10, 1])

# python -m src.model.vq_mh


        
