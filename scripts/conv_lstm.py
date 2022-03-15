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

device = "cuda"

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, mode="zeros"):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.mode = mode

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        x = x.to(device)
        h_cur = h_cur.to(device)
        
        concat_input_hcur = torch.cat([x, h_cur], dim=1) 
        concat_input_hcur = concat_input_hcur.to(device)

        concat_input_hcur_conv = self.conv(concat_input_hcur)
        concat_input_hcur_conv = concat_input_hcur_conv.to(device)

        cc_input_gate, cc_forget_gate, cc_output_gate, cc_output = torch.split(concat_input_hcur_conv, self.hidden_dim, dim=1)
        
        input_gate = torch.sigmoid(cc_input_gate +  c_cur)

        forget_gate = torch.sigmoid(cc_forget_gate +  c_cur)

        output = torch.tanh(cc_output)

        c_next = forget_gate * c_cur + input_gate * output

        output_gate = torch.sigmoid(cc_output_gate +  c_next)

        h_next = output * torch.tanh(c_next)

        return h_next, c_next

    def init_state(self, batch_size, image_size):
        height, width = image_size
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        elif(self.mode == "random"):
            h = torch.randn(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
            c = torch.randn(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        elif(self.mode == "learned"):
            h = self.learned_h.repeat(batch_size, 1, height, width, device=self.conv.weight.device)
            c = self.learned_c.repeat(batch_size, 1, height, width, device=self.conv.weight.device)
        
        return h, c

        
class ConvLSTM(nn.Module):
    """ 
    Custom LSTM for images. Batches of images are fed to a Conv LSTM
    
    Args:
    -----
    input_dim: integer
        Number of channels of the input.
    hidden_dim: integer
        dimensionality of the states in the cell
    kernel_size: tuple
        size of the kernel for convolutions
    num_layers: integer
        number of stacked LSTMS
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
       
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        conv_lstms  = []
        # iterating over no of layers
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            conv_lstms.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.conv_lstms = nn.ModuleList(conv_lstms)

    def forward(self, x, hidden_state=None):

        b, frames, c, h, w = x.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
        cur_layer_input = x

        
        # iterating over no of layers
        for i in range(self.num_layers):

            h, c = hidden_state[i]
            each_layer_output = []
            # iterating over sequence length

            for t in range(frames*2):
                if t < 10:
                    h, c = self.conv_lstms[i](x=cur_layer_input[:, t, :, :, :],cur_state=[h, c])
                
                elif t >= 10:
                    h_prev = each_layer_output[-1]
                    h, c = self.conv_lstms[i](x = h_prev, cur_state=[h, c])
                
                each_layer_output.append(h)

            stacked_layer_output = torch.stack(each_layer_output, dim=1)
        return stacked_layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.conv_lstms[i].init_state(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param