import pandas as pd
import numpy as np

def load_data(file_path:str):
    """Load data from an Excel file."""
    data = pd.read_csv(file_path)
    data = data[['time(h)', 'formaldehyde(ug/m³)']]
    # data = data[['formaldehyde(ug/m³)']]
    return data

def normalize_data(data:pd.DataFrame):
    """Preprocess the data by normalizing it."""
    data = (data - data.mean()) / data.std()
    return data

def prepare_sequences(data:pd.DataFrame, time_steps:int=1):
    """Prepare sequences for LSTM input."""
    sequences = []
    labels = []
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    for i in range(len(data) - time_steps):
        seq = data[i:(i + time_steps)]
        label = data[i + time_steps, 1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def load_and_preprocess_data(file_path, time_steps:int=1):
    """Main function to load, and prepare data for training."""
    return prepare_sequences(normalize_data(load_data(file_path)), time_steps)

if __name__ == '__main__':
    w_plus_y_sequences, w_plus_y_labels = load_and_preprocess_data("data/w+y.csv", 10)
    y_plus_w_sequences, y_plus_w_labels = load_and_preprocess_data("data/y+w.csv", 10)
    print(w_plus_y_sequences.shape, w_plus_y_labels.shape)
    print(y_plus_w_sequences.shape, y_plus_w_labels.shape)

# python -m src.utils.data2d