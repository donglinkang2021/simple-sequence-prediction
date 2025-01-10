import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_pe, split_heads, cat_heads

class Model(nn.Module):
    def __init__(
        self, 
        d_model: int,
        n_heads: int,
        kv_heads: int,
        input_dim: int,
        output_dim: int,
        block_size: int = 512,
        # kwargs to find which architecture is better
        is_inln: bool = False,  # whether apply layer norm to input
        is_qln: bool = False,   # whether apply layer norm to Q
        is_kln: bool = False,   # whether apply layer norm to K
        is_causal: bool = False, # whether add causal mask
        is_n_pe: bool = False,  # whether apply pe after Input or Q, K
        pe_type: str = None,    # 'randn', 'sinpe', 'rope', None
    ):
        super(Model, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % kv_heads == 0, "n_heads must be divisible by kv_heads"
        head_dim = d_model // n_heads
        self.enable_gqa = n_heads > kv_heads
        self.head_dim = head_dim
        
        self.is_inln = is_inln
        self.is_qln = is_qln
        self.is_kln = is_kln
        self.is_causal = is_causal
        self.is_n_pe = is_n_pe
        
        # Input projection
        self.fc_in = nn.Linear(input_dim, d_model)
        if is_inln:
            self.ln_in = nn.LayerNorm(d_model)
            
        # Positional encoding
        self.pe = get_pe(pe_type, head_dim if is_n_pe else d_model, block_size)
        
        # Layer norms for Q, K
        if is_qln:
            self.ln_q = nn.LayerNorm(head_dim)
        if is_kln:
            self.ln_k = nn.LayerNorm(head_dim)
            
        # QKV projection and output
        self.fc_q = nn.Linear(d_model, d_model, bias=False)
        self.fc_k = nn.Linear(d_model, kv_heads * head_dim, bias=False)
        self.fc_v = nn.Linear(d_model, kv_heads * head_dim, bias=False)
        self.fc_out = nn.Linear(d_model, output_dim, bias=False)

    def forward(self, x):
        # input x shape: (B, T, input_dim)
        x = self.fc_in(x)
        
        if not self.is_n_pe and self.pe is not None:
            x = self.pe(x)

        if self.is_inln:
            x = self.ln_in(x)
            
        xq, xk, xv = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        xq, xk, xv = map(lambda x: split_heads(x, self.head_dim), (xq, xk, xv))

        if self.is_n_pe and self.pe is not None:
            xq = self.pe(xq)
            xk = self.pe(xk)
        
        if self.is_qln:
            xq = self.ln_q(xq)
        if self.is_kln:
            xk = self.ln_k(xk)
            
        xo = F.scaled_dot_product_attention(xq, xk, xv, is_causal=self.is_causal, enable_gqa=self.enable_gqa)
        xo = cat_heads(xo)
        return self.fc_out(xo)

if __name__ == "__main__":
    from .utils import init_weights, model_summary
    import numpy as np
    
    B, T, D = 32, 10, 1
    config = dict(
        d_model=64,
        n_heads=8,
        kv_heads=2,
        input_dim=D,
        output_dim=1,
        block_size=218,
        is_inln=True,
        is_qln=True,
        is_kln=True,
        is_causal=True,
        is_n_pe=True,
        pe_type='rope'
    )
    
    model = Model(**config)
    model.apply(init_weights)
    model_summary(model)

    seq = np.random.randn(B, T, D).astype(np.float32)
    seq = torch.from_numpy(seq)
    out = model(seq)
    print(out.shape)  # Expected output: torch.Size([32, 10, 1])

# python -m src.model.att_mh