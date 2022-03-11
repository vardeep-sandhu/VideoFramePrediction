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
from dataset import MNIST_Moving
from encoder import FirstLayerEncBlock

class Model(nn.Module):
    enc_chs_1 = (1,16,64)
    dec_chs_1 = (64,16,1)
    
    enc_layer = FirstEncoder(enc_chs)
    # This encoder for the first layer gives values [1, 64, 16, 16]

    conv_lstm_layer = ConvLSTM(input_dim= 64, hidden_dim = 64, kernel_size = (5,5), num_layers= 1)

    decoder_layer_1 = FirstDecoder(dec_chs)
    # The encoder decoder setup gives correct dimensional results