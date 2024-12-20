import torch
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra
from torch.utils.tensorboard import SummaryWriter

from src.data import get_data_loaders
from src.utils.logger import omegaconf_dict_to_tb_hparams
from src.model.utils import init_weights
from src.train import train_epoch, eval_epoch

def setup(cfg: DictConfig):
    log_dir = f"{cfg.logger.save_dir}/{cfg.logger.name}/{cfg.logger.version}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = SummaryWriter(log_dir)    
    return log_dir, logger

@hydra.main(config_path='conf', config_name='default', version_base=None)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    # return
    log_dir, logger = setup(cfg)
    num_epochs = cfg.train.num_epochs
    learning_rate = cfg.train.learning_rate
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        cfg.dataset.train_file, 
        cfg.train.time_steps, 
        cfg.train.batch_size, 
        cfg.train.is_random_split,
        cfg.train.is_align_target
    )

    # Initialize model, criterion, and optimizer
    model = hydra.utils.instantiate(cfg.model)
    model.apply(init_weights)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    pbar = tqdm(total=num_epochs, desc='Training', unit='step', dynamic_ncols=True)
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        logger.add_scalar('Loss/train', train_loss, epoch)
        logger.add_scalar('Loss/val', val_loss, epoch)
        pbar.set_postfix(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss)
        pbar.update()
    pbar.close()

    # Log hyperparameters and metrics
    logger.add_hparams(
        hparam_dict = omegaconf_dict_to_tb_hparams(cfg),
        metric_dict = {
            'Final/train_loss': train_loss,
            'Final/val_loss': val_loss
        },
        run_name='hparams'
    )

    # Save the trained model
    torch.save(model.state_dict(), f'{log_dir}/model.pth')
    # Save the config
    OmegaConf.save(cfg, f'{log_dir}/config.yaml')
    
if __name__ == '__main__':
    main()

# python train.py