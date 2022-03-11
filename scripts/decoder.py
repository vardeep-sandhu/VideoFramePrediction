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


class FirstLayerDecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 3,stride=2,padding=(1, 1))
        self.conv2 = nn.ConvTranspose2d(out_ch, out_ch, 2,stride=1)
    
    def forward(self, x):
        return self.conv2((self.conv1(x)))

class FirstDecoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.dec_blocks = nn.ModuleList([FirstLayerDecBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        
    
    def forward(self, x):
        layers = []
        print("Starting decoding step")
        print("Input shape", x.shape)

        for block in self.dec_blocks:
            x = block(x)
            print("Intermediate shape", x.shape)   
        return x