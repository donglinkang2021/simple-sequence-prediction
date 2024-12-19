import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(predictions, seq, save_path='results.png'):
    """
    Plot the actual and predicted values.
    seq: numpy array of shape (T, 2)
    predictions: numpy array of shape (n_steps,)
    """
    n_steps = len(predictions)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(14, 8))
    
    plt.plot(seq[:, 0], seq[:, 1], label='Original Data (Complete)', color='blue', linewidth=2) 
    plt.plot(seq[-n_steps:, 0], predictions, label='Predicted Values (Test Set)', color='red', linestyle='--', linewidth=2)
    plt.plot(seq[-n_steps:, 0], seq[-n_steps:, 1], label='Actual Values (Test Set)', color='green', linestyle='dotted', linewidth=3)
    plt.axvline(x=seq[-n_steps, 0], color='black', linestyle='--', label='Prediction Start', linewidth=2)
    
    plt.title('Original Dataset, Test Set True Values, and Predictions', fontsize=18)
    plt.xlabel('Time (h)', fontsize=15)
    plt.ylabel('Formaldehyde Concentration (ug/m³)', fontsize=15)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=200)
    print(f"Plot saved at {save_path}")
