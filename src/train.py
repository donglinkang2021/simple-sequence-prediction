import torch
import torch.nn as nn

def train_epoch(
        model:nn.Module, 
        train_loader:torch.utils.data.DataLoader, 
        criterion:nn.Module, 
        optimizer:torch.optim.Optimizer, 
        device:torch.device,
    ):
    model.train()
    total_loss = 0
    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def eval_epoch(
        model:nn.Module, 
        val_loader:torch.utils.data.DataLoader, 
        criterion:nn.Module, 
        device:torch.device,
    ):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs, _ = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)
