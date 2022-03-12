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


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 3,stride=2,padding=(1, 1))
        self.conv2 = nn.ConvTranspose2d(out_ch, out_ch, 2,stride=1)
    
    def forward(self, x):
        return self.conv2((self.conv1(x)))

class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.dec_blocks = nn.ModuleList([DecBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        
    
    def forward(self, x):
        layers = []
        # print("hello")
        # print(x.shape)
        for block in self.dec_blocks:
            x = block(x)
            # print(x.shape)
            layers.append(x)

        return x

class Embedded_Decoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        #The decoder is in reverse here 
        self.firstDecoder= Decoder((256,128))
        self.firstDecoder.to(device)
        self.secondDecoder= Decoder((256,64))
        self.secondDecoder.to(device)
        self.thirdDecoder= Decoder((128,16,1))
        self.thirdDecoder.to(device)

    def forward(self, x):
      #list to store the outputs from each decoders
        decoder_outputs = []
        x=self.firstDecoder(x)
        decoder_outputs.append(x)
        x=self.secondDecoder(x)
        decoder_outputs.append(x)
        x=self.thirdDecoder(x)
        decoder_outputs.append(x)
        
        return decoder_outputs