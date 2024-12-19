import torch
from torch.utils.data import Dataset, DataLoader, random_split
from .utils.data import load_and_preprocess_data

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

        # if target is 1D, add a dimension
        if len(self.targets.shape) == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def get_data_loaders(train_file:str, time_steps:int, batch_size:int, is_random_split:bool=True):
    # Load and preprocess data
    sequences, targets = load_and_preprocess_data(train_file, time_steps)
    num_sameples = len(sequences)
    # Create dataset and split into training and validation sets
    
    n_train = int(0.8 * num_sameples)

    if is_random_split:
        train_dataset, val_dataset = random_split(
            SequenceDataset(sequences, targets), 
            [n_train, num_sameples - n_train],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = SequenceDataset(sequences[:n_train], targets[:n_train])
        val_dataset = SequenceDataset(sequences[n_train:], targets[n_train:])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == '__main__':
    train_file = 'data/w+y.csv'
    time_steps = 10
    batch_size = 32
    train_loader, val_loader = get_data_loaders(train_file, time_steps, batch_size)
    for sequences, targets in train_loader:
        print(sequences.shape, targets.shape)
        break
    for sequences, targets in val_loader:
        print(sequences.shape, targets.shape)
        break

# python -m src.data