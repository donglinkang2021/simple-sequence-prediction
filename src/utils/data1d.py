import pandas as pd
import numpy as np

def load_data(file_path:str) -> np.ndarray:
    """Load data from an Excel file."""
    data = pd.read_csv(file_path)
    return data['formaldehyde(ug/m³)'].to_numpy()

def prepare_sequences(data:np.ndarray, time_steps:int=1):
    """Prepare sequences for LSTM input."""
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:(i + time_steps)])
        labels.append(data[i + time_steps])
    return np.array(sequences), np.array(labels)

def load_and_preprocess_data(file_path, time_steps:int=1):
    data = load_data(file_path)
    data = (data - data.mean()) / data.std()
    return prepare_sequences(data, time_steps)

if __name__ == '__main__':
    w_plus_y_sequences, w_plus_y_labels = load_and_preprocess_data("data/w+y.csv", 10)
    y_plus_w_sequences, y_plus_w_labels = load_and_preprocess_data("data/y+w.csv", 10)
    print(w_plus_y_sequences.shape, w_plus_y_labels.shape)
    print(y_plus_w_sequences.shape, y_plus_w_labels.shape)

# python -m src.utils.data1d