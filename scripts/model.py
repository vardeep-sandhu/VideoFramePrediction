import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from dataset import MNIST_Moving
from encoder import Embedded_Encoder
from decoder import Embedded_Decoder
from conv_lstm import ConvLSTM


device = "cuda"

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.embd_model = Embedded_Encoder(device)

        self.conv_model_FirstEncoder = ConvLSTM(input_dim= 64, hidden_dim = 64, kernel_size = (5,5), num_layers= 2)
        if torch.cuda.is_available():
            self.conv_model_FirstEncoder.to(device)

        self.conv_model_SecondEncoder= ConvLSTM(input_dim= 128, hidden_dim = 128, kernel_size = (5,5), num_layers= 2)
        if torch.cuda.is_available():
            self.conv_model_SecondEncoder.to(device)

        self.conv_model_ThirdEncoder= ConvLSTM(input_dim= 256, hidden_dim = 256, kernel_size = (5,5), num_layers= 2)
        if torch.cuda.is_available():
            self.conv_model_ThirdEncoder.to(device)

        self.decoder = Embedded_Decoder(device)


    def forward(self, x):
                
        embds = self.embd_model(x)
        
        lstm_1, (h_1, c_1) = self.conv_model_FirstEncoder(embds[0])
        lstm_2, (h_2, c_2) = self.conv_model_SecondEncoder(embds[1])
        lstm_3, (h_3, c_3) = self.conv_model_ThirdEncoder(embds[2])

        
        out_1 = self.decoder.firstDecoder(lstm_3.squeeze())
        in_2 = torch.cat((out_1, lstm_2.squeeze()), 1)

        out_2 = self.decoder.secondDecoder(in_2)
        in_3 = torch.cat((out_2, lstm_1.squeeze()), 1)

        out_3 = self.decoder.thirdDecoder(in_3)
        return out_3