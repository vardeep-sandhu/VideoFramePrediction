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


# This model is inspired by ResNet
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int =
1, dilation: int = 1):
   """3x3 convolution with padding"""
   return nn.ConvTranspose2d(
   in_planes,
   out_planes,
   kernel_size=3,
   stride=stride,
   padding=dilation,
   groups=groups,
   bias=False,
   dilation=dilation,
   )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
   """1x1 convolution"""
   return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2,
stride=stride,bias=False)


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_ch = out_ch 
        self.conv1 = conv3x3(in_planes=in_ch, out_planes=in_ch)
        self.bn1  = nn.BatchNorm2d(in_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes=in_ch, out_planes=in_ch)
        self.bn2  = nn.BatchNorm2d(in_ch)
        # This 1 X 1 layer downsamples and inc channel as required
        self.conv3 = conv1x1(in_planes=in_ch, out_planes=out_ch, stride=2)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        print("output channel", self.out_ch)
        print("decoder shape afte 2 convs",out.shape)
        # Till now channel and spatial dims of out and identity are same
        # Now we inc channel length
        out = self.conv3(out)
        out = self.bn3(out)
        print("decoder shape afte 3 convs",out.shape)
        return out

class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.dec_blocks = nn.ModuleList([DecBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        
    
    def forward(self, x):
        for block in self.dec_blocks:
            x = block(x)
        return x

class Embedded_Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        #The decoder is in reverse here 
        self.firstDecoder= Decoder((256,128))

        self.secondDecoder= Decoder((256,64))

        self.thirdDecoder= Decoder((128,64,1))

        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, lstm_outputs, batch_size, seq_len):

        # Decoding all images together so changed dims to [Batch_size * t, 256, 4, 4]
        batch_size, seq_len = lstm_outputs[0].shape[0:2]

        first_decoder_out = self.firstDecoder(lstm_outputs[0].view(-1, *lstm_outputs[0].shape[2:]))
        print("fd shape", first_decoder_out.shape)
        l1=lstm_outputs[1].view(-1, *lstm_outputs[1].shape[2:])
        print("l1 shape", l1.shape)

        first_dec_with_lstm = torch.cat((first_decoder_out, lstm_outputs[1].view(-1, *lstm_outputs[1].shape[2:])), 1)

        scnd_dec_output = self.secondDecoder(first_dec_with_lstm)

        scnd_dec_output_with_lstm = torch.cat((scnd_dec_output, lstm_outputs[2].view(-1, *lstm_outputs[2].shape[2:])), 1)
        

        out_3 = self.thirdDecoder(scnd_dec_output_with_lstm)
        output = out_3.reshape(batch_size, seq_len, *out_3.shape[1:])
        return self.sigmoid_layer(output)
