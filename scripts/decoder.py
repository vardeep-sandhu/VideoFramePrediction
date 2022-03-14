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
        self.elu  = nn.ELU()
        self.bn1    = nn.BatchNorm2d(out_ch)
        
    
    def forward(self, x):
        return self.elu(self.bn1(self.conv2(self.elu(self.bn1(self.conv1(x))))))

class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.dec_blocks = nn.ModuleList([DecBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        
    
    def forward(self, x):
        layers = []
        
        for block in self.dec_blocks:
            x = block(x)
            layers.append(x)

        return x

class DecoderDimensionMatch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return x

class Embedded_Decoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        #The decoder is in reverse here 
        self.firstDecoder= Decoder((256,128))
        self.firstDecoder.to(device)

        # Add DecoderDimensionMatch block
        self.chnge_dims_256_128 = DecoderDimensionMatch(256, 128)
        self.chnge_dims_256_128.to(device)

        self.secondDecoder= Decoder((128,64))
        self.secondDecoder.to(device)

        # Add DecoderDimensionMatch block
        self.chnge_dims_128_64 = DecoderDimensionMatch(128, 64)
        self.chnge_dims_128_64.to(device)

        self.thirdDecoder= Decoder((64,16,1))
        self.thirdDecoder.to(device)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
#       List to store the outputs from each decoders
        
        decoder_outputs = []
        x=self.firstDecoder(x)
        decoder_outputs.append(x)
        x=self.secondDecoder(x)
        decoder_outputs.append(x)
        x=self.thirdDecoder(x)
        decoder_outputs.append(x)

        return self.sigmoid_layer(decoder_outputs)