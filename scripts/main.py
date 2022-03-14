import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import MNIST_Moving
from model import Model
import utils

if __name__ == "__main__":
    # Adding dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    #transforms are here!
    
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((64, 64)),
                        transforms.Normalize((0.5, ), (0.5, ))
                       ])
    # [TODO]
    # And visualize the input images using the vis function

    train_set = MNIST_Moving(root='.data/mnist', train=True, download=True, transform=transform, target_transform=transform)
    test_set = MNIST_Moving(root='.data/mnist', train=False, download=True, transform=transform, target_transform=transform)
    
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=False)

    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    print("Data loaders ready")

    model = Model()
    
    
    print("Model Loaded")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # mean_loss, loss_list = utils.train_epoch(model, train_loader, optimizer, criterion, 0, device)
    # breakpoint()
    
    train_loss, test_loss, loss_iter, epochs = utils.train_model(model, optimizer, scheduler, criterion,\
                                                                train_loader, test_loader, 100, device)