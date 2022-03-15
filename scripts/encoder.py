import torch
import torch.nn as nn

# This model is inspired by ResNet 
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )



class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_planes=in_ch, out_planes=out_ch)
        self.bn1    = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(in_planes=out_ch, out_planes=out_ch)
        self.bn2    = nn.BatchNorm2d(out_ch)
        
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.shape)
        # print(identity.shape)
        # out += identity
        out = self.relu(out)

        return out

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.enc_blocks = nn.ModuleList([EncBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        identity = x

        for block in self.enc_blocks:
            # print("block 1")
            x = block(x)
            # print(x.shape)
        # Residual connection from input
        # x += identity
            x = self.pool(x)
        return x


class Embedded_Encoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.firstEncoder= Encoder((1,16,64))
        self.firstEncoder.to(device)
        self.secondEncoder= Encoder((64,128))
        self.secondEncoder.to(device)
        self.thirdEncoder= Encoder((128,256))
        self.thirdEncoder.to(device)

    def forward(self, x):
        # Embedder makes embeddings for all the frames in all batch sequences
        input_shape = x.shape
        
        x = x.reshape(-1, *input_shape[2:])
        #list to store the outputs from each encoders
        encoder_outputs = []
        # print("first encoder not done")

        x=self.firstEncoder(x)
        # print("first encoder done ")
        encoder_outputs.append(x)
        x=self.secondEncoder(x)
        encoder_outputs.append(x)
        x=self.thirdEncoder(x)
        encoder_outputs.append(x)


        for idx, emds in enumerate(encoder_outputs):
            emds_dims = emds.shape[1:]
            encoder_outputs[idx] = emds.reshape(input_shape[0], 10, *emds_dims)
            
        return encoder_outputs