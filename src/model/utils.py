import torch
import torch.nn as nn
from tabulate import tabulate
from simplepe import *
from einops import rearrange

def split_heads(x:torch.Tensor, head_size:int):
    return rearrange(x, 'B T (nH Hs) -> B nH T Hs', Hs=head_size)

def cat_heads(x:torch.Tensor):
    return rearrange(x, 'B nH T Hs -> B T (nH Hs)')

def get_pe(pe_type, d_model, block_size):
    if pe_type == 'randn':
        return RandomNormalPositionalEncoding(d_model, max_len=block_size)
    elif pe_type == 'sinpe':
        return SinusoidalPositionalEncoding(d_model, max_len=block_size)
    elif pe_type == 'rope':
        return RotaryPositionalEncoding(d_model, max_len=block_size)
    return None

def init_weights(module: nn.Module):
    """usage: model.apply(init_weights)"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: nn.Module):
    num_params = count_parameters(model)
    metrics = {
        'model': model.__class__.__name__,
        'package': model.__class__.__module__,  # Added package name
        'num_params': num_params,
        'size(MB)': num_params * 4 / 1024 / 1024
    }
    table = tabulate([metrics], headers='keys', tablefmt='pretty')
    print(table)
