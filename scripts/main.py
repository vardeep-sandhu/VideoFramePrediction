import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import MNIST_Moving
from model import Model
import utils
import wandb

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((64, 64)),
                       ])
    
    train_set = MNIST_Moving(root='.data/mnist', train=True, download=True, transform=transform, target_transform=transform)
    test_set = MNIST_Moving(root='.data/mnist', train=False, download=True, transform=transform, target_transform=transform)
    
    batch_size = 32

    train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=False)

    test_loader = DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    print("Data loaders ready")

# W and B for logging grads
    wandb.init()
    
    model = Model()
    mdoel = model.to(device)
    wandb.watch(model, log_freq=100)

    print("Model Loaded")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    # mean_loss, loss_list = utils.train_epoch(model, train_loader, optimizer, criterion, 0, device)

    train_loss, test_loss, loss_iter, epochs = utils.train_model(model, optimizer, scheduler, criterion,\
                                                                train_loader, test_loader, 100, device)
