import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class FirstLayerEncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Add BN here 
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,stride=1,padding=(1, 1))
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,stride=1,padding=(1, 1))
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class FirstEncoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([FirstLayerEncBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        layers = []
        print("Staring encoding process")
        print("Input to the encoder", x.shape)
        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)
            print("Output after block")
            print(x.shape)
            layers.append(x)
        return x

