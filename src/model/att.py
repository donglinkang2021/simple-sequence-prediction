import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_pe

class Model(nn.Module):
    def __init__(
        self, 
        d_model: int,
        input_dim: int,
        output_dim: int,
        block_size: int = 512,
        is_inln: bool = False,
        is_qln: bool = False,
        is_kln: bool = False,
        is_causal: bool = False,
        is_n_pe: bool = False,  # whether apply pe after Input or Q, K
        pe_type: str = None,  # 'randn', 'sinpe', 'rope', None
    ):
        super(Model, self).__init__()

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
        self.pe = get_pe(pe_type, d_model, block_size)
        
        # Layer norms for Q, K
        if is_qln:
            self.ln_q = nn.LayerNorm(d_model)
        if is_kln:
            self.ln_k = nn.LayerNorm(d_model)
            
        # QKV projection and output
        self.fc_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc_out = nn.Linear(d_model, output_dim, bias=False)

    def forward(self, x):
        # input x shape: (B, T, input_dim)
        x = self.fc_in(x)
            
        if not self.is_n_pe and self.pe is not None:
            x = self.pe(x)
        
        if self.is_inln:
            x = self.ln_in(x)    

        qkv = self.fc_qkv(x)
        xq, xk, xv = qkv.chunk(chunks=3, dim=-1)

        if self.is_n_pe and self.pe is not None:
            xq = self.pe(xq)
            xk = self.pe(xk)
        
        if self.is_qln:
            xq = self.ln_q(xq)
        if self.is_kln:
            xk = self.ln_k(xk)
            
        xo = F.scaled_dot_product_attention(xq, xk, xv, is_causal=self.is_causal)
        return self.fc_out(xo)

if __name__ == "__main__":
    from .utils import init_weights, model_summary
    import numpy as np
    
    B, T, D = 32, 10, 1
    config = dict(
        d_model=64,
        input_dim=D,
        output_dim=1,
        block_size=218,
        is_inln=True,
        is_qln=True,
        is_kln=True,
        is_causal=True,
        pe_type='rope'
    )
    
    model = Model(**config)
    model.apply(init_weights)
    model_summary(model)

    seq = np.random.randn(B, T, D).astype(np.float32)
    seq = torch.from_numpy(seq)
    out = model(seq)
    print(out.shape)  # Expected output: torch.Size([32, 10, 1])

# python -m src.model.att