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

class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,stride=1,padding=(1, 1))
        self.elu  = nn.ELU()
        self.bn1    = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,stride=1,padding=(1, 1))
    
    def forward(self, x):
        return self.elu(self.bn1(self.conv2(self.elu(self.bn1(self.conv1(x))))))

class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([EncBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        layers = []

        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)
            layers.append(x)
        return x


class Embedded_Encoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.firstEncoder= Encoder((1,16,64))
        # self.firstEncoder.to(device)
        self.secondEncoder= Encoder((64,128))
        # self.secondEncoder.to(device)
        self.thirdEncoder= Encoder((128,256))
        # self.thirdEncoder.to(device)

    def forward(self, x):
        # Embedder makes embeddings for all the frames in all batch sequences
        input_shape = x.shape
        
        x = x.reshape(-1, *input_shape[2:])
        #list to store the outputs from each encoders
        encoder_outputs = []
        x=self.firstEncoder(x)
        encoder_outputs.append(x)
        x=self.secondEncoder(x)
        encoder_outputs.append(x)
        x=self.thirdEncoder(x)
        encoder_outputs.append(x)


        for idx, emds in enumerate(encoder_outputs):
            emds_dims = emds.shape[1:]
            encoder_outputs[idx] = emds.reshape(input_shape[0], 10, *emds_dims)
            
        return encoder_outputs