import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from warmup import ReduceLROnPlateauWithWarmup

from model import Model
import utils
import wandb
import argparse
from attrdict import AttrDict
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup 

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', '-c', type=str)
    parser.add_argument('--scheduler','-s', type=str)
    args = parser.parse_args()
    cfg = AttrDict(utils.load_cfg(args.cfg_name))
    return args, cfg

warmup_epochs = 3
warmup_initial_lr = 0.004

if __name__ == "__main__":
    
    args, cfg = parse_commandline()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((64, 64)),
                       ])

    train_set, test_set = utils.load_dataset(cfg)
    
    batch_size = cfg.batch_size

    train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)

    test_loader = DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=True)
    print("Data loaders ready")

# W and B for logging grads
    wandb.init()
    
    model = Model()
    mdoel = model.to(device)
    wandb.watch(model, log_freq=cfg.log_freq)

    print("Model Loaded")
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr= cfg.learning_rate)
    if args.scheduler == "plateau":
       scheduler = ReduceLROnPlateauWithWarmup(optimizer,0,cfg.learning_rate,10)
    elif args.scheduler == "exponential":
       exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
       scheduler = create_lr_scheduler_with_warmup(exp_scheduler,
                                            warmup_start_value=cfg.warmup_initial,
                                            warmup_end_value=cfg.learning_rate,
                                            warmup_duration=cfg.warmup_epochs)
    # mean_loss, loss_list = utils.train_epoch(model, train_loader, optimizer, criterion, 0, device)

    train_loss, test_loss, loss_iter, epochs = utils.train_model(model, optimizer, scheduler, criterion,\
                                                                train_loader, test_loader, cfg.epochs, device)
