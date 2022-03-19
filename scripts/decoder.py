from typing import final
import torch
import torch.nn as nn


# This model is inspired by ResNet
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
   """3x3 transposed convolution with padding"""
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
   return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_ch = out_ch 
        self.conv1 = conv3x3(in_planes=in_ch, out_planes=in_ch)
        self.bn1  = nn.BatchNorm2d(in_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes=in_ch, out_planes=in_ch)
        self.bn2  = nn.BatchNorm2d(in_ch)
        
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
        
        out = self.conv3(out)
        out = self.bn3(out)

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
        self.firstDecoder= Decoder((256, 128))

        self.secondDecoder= Decoder((128, 64))

        self.thirdDecoder= Decoder((64, 16, 1))

        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, lstm_outputs, final_enc_embeddings , batch_size, seq_len):

        # Decoding all images together so changed dims to [Batch_size * t, 256, 4, 4]
        batch_size, seq_len = lstm_outputs[0].shape[0:2]

        last_context_frame = final_enc_embeddings[:, -1, :, :, :].unsqueeze(1)
        last_context_repeat = torch.cat([last_context_frame]*10, dim=1)        
        final_enc_embeddings = torch.cat((final_enc_embeddings, last_context_repeat), dim=1)

        for idx, _ in enumerate(lstm_outputs):
            lstm_outputs[idx] = lstm_outputs[idx].reshape(-1, *lstm_outputs[idx].shape[2:])

        
        fst_dec = self.firstDecoder(lstm_outputs[0] + final_enc_embeddings.reshape(*lstm_outputs[0].shape)) 
        fst_dec = self.firstDecoder(lstm_outputs[0])
        scnd_dec = self.secondDecoder(fst_dec + lstm_outputs[1]) 
        thrd_dec = self.thirdDecoder(scnd_dec + lstm_outputs[2])
        out = self.sigmoid_layer(thrd_dec)
        
#       Reshape to 5D again
        output = out.reshape(batch_size, seq_len, *out.shape[1:])

        return output