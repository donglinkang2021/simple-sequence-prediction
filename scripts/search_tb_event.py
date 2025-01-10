import os
import glob
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator
from tabulate import tabulate
from omegaconf import OmegaConf

def get_run_name(file: str) -> str:
    return os.path.join(*file.split('/')[:4])

def get_model_params(run_name: str) -> dict:
    cfg = OmegaConf.load(os.path.join(run_name, 'config.yaml'))
    return cfg.model

def model_params_to_str(model_params: dict) -> str:
    model_short = model_params['_target_'].split('.')[-2]
    model_params.pop('_target_')
    for k, v in model_params.items():
        if isinstance(v, bool):
            model_short += f'-{k}' if v else ''
        else:
            model_short += f'-{v}'
    return model_short

def main():
    log_dir = './logs'
    # log_dir = '.cache'

    # metrics = { 'file0': { 'metric1' : 0.1, 'metric2': 0.2, ...}, ...}
    metrics = []
    
    event_files = glob.glob(os.path.join(log_dir, '**/hparams/events.out.tfevents*'), recursive=True)
    
    # def search_condition(x):
    #     return 'att' in x
    # event_files = list(filter(search_condition, event_files))

    for file in tqdm(event_files, desc="Processing event files", dynamic_ncols=True):
        ea = event_accumulator.EventAccumulator(file)
        ea.Reload()
        run_name = get_run_name(file)
        model_parms = get_model_params(run_name)
        model_short = model_params_to_str(model_parms)
        metric = {'Model': f"[{model_short}]({run_name})"}
        for tag in ea.Tags().get('scalars', []):
            # each ea.Scalars(tag) just contains one scalar
            metric[tag] = ea.Scalars(tag)[0].value
        metrics.append(metric)
        
    print(f"Found {len(metrics)} event files.")

    tablefmt = 'github'
    # tablefmt = 'fancy_grid'
    sorted_metrics = sorted(metrics, key=lambda x: x['Predict/mse_regressive'])
    # print(tabulate(sorted_metrics[:10], headers='keys', tablefmt=tablefmt))
    print(tabulate(sorted_metrics[:20], headers='keys', tablefmt=tablefmt))

if __name__ == '__main__':
    main()
