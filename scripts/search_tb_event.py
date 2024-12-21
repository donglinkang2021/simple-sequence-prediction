import os
import glob
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator
from tabulate import tabulate

def main():
    log_dir = './logs'
    loss_values = []
    
    event_files = glob.glob(os.path.join(log_dir, '**/hparams/events.out.tfevents*'), recursive=True)
    
    for file in tqdm(event_files, desc="Processing event files", dynamic_ncols=True):
        ea = event_accumulator.EventAccumulator(file)
        ea.Reload()
        for tag in ea.Tags().get('scalars', []):
            if 'Final/val_loss' in tag:
                for event in ea.Scalars(tag):
                    loss_values.append(event.value)
    
    min_loss = min(loss_values)
    num_files = len(event_files)
    metrics = {
        'min_loss': min_loss,
        'num_files': num_files,
    }
    tablefmt = 'github'
    # tablefmt = 'fancy_grid'
    print(tabulate([metrics], headers='keys', tablefmt=tablefmt))

    def get_run_name(file):
        splits = file.split('/')
        return splits[2] + '/' + splits[3]

    sorted_file_loss = sorted(zip(map(get_run_name, event_files), loss_values), key=lambda x: x[1])
    print(tabulate(sorted_file_loss[:10], headers=['File', 'Loss'], tablefmt=tablefmt))

if __name__ == '__main__':
    main()
