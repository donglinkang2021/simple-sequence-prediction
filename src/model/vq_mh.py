import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_pe, split_heads, cat_heads

class Model(nn.Module):
    def __init__(
            self, 
            vocab_size:int, 
            d_model:int, 
            n_heads:int,
            kv_heads: int,
            input_dim:int, 
            output_dim:int, 
            block_size:int, 
            is_inln: bool = False,
            is_qln: bool = False,
            is_n_pe: bool = False,
            is_causal: bool = False,
            pe_type: str = None,  # 'randn', 'sinpe', 'rope', None
        ):
        super(Model, self).__init__()
        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        assert n_heads % kv_heads == 0, "n_heads must be divisible by kv_heads"
        head_dim = d_model // n_heads
        self.enable_gqa = n_heads > kv_heads
        self.head_dim = head_dim

        self.is_inln = is_inln
        self.is_qln = is_qln
        self.is_causal = is_causal
        self.is_n_pe = is_n_pe

        self.fc_in = nn.Linear(input_dim, d_model)
        
        if is_inln:
            self.ln_in = nn.LayerNorm(d_model)

        self.pe = get_pe(pe_type, head_dim if is_n_pe else d_model, block_size)

        if is_qln:
            self.ln_q = nn.LayerNorm(head_dim)

        self.cache = nn.Parameter(
            torch.randn(kv_heads, vocab_size, head_dim), requires_grad=True
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # input x shape: (B, T, 1)
        x = self.fc_in(x) # -> (B, T, D)

        if not self.is_n_pe and self.pe is not None:
            x = self.pe(x)

        if self.is_inln:
            x = self.ln_in(x)

        x = split_heads(x, self.head_dim) # -> (B, nH, T, Hs)
        
        if self.is_n_pe and self.pe is not None:
            x = self.pe(x)
        
        if self.is_qln:
            x = self.ln_q(x)

        xo = F.scaled_dot_product_attention(x, self.cache, self.cache, is_causal=self.is_causal, enable_gqa=self.enable_gqa)
        xo = cat_heads(xo)
        return self.fc_out(xo)

if __name__ == "__main__":
    from .utils import init_weights, model_summary
    import numpy as np
    B, T, D = 32, 10, 1
    model = Model(
        vocab_size=32, 
        d_model=64, 
        n_heads=8,
        kv_heads=2, 
        input_dim=D, 
        output_dim=1, 
        block_size=218, 
        is_inln=True,
        is_qln=True,
        is_causal=True,
        is_n_pe=True,
        pe_type='rope'
    )
    model.apply(init_weights)
    model_summary(model)

    seq = np.random.randn(B, T, D).astype(np.float32)
    seq = torch.from_numpy(seq)
    out = model(seq)
    print(out.shape)  # Expected output: torch.Size([32, 10, 1])

# python -m src.model.vq_mh


        
