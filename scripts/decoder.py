import torch
import torch.nn as nn




class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_ch = out_ch
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, kernel_size = 3,stride=2,padding=(1, 1))
        self.conv2 = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
    
    def forward(self, x):
        if self.out_ch == 1:
            return self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        else:
            return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))

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

        first_dec_with_lstm = torch.cat((first_decoder_out, lstm_outputs[1].view(-1, *lstm_outputs[1].shape[2:])), 1)

        scnd_dec_output = self.secondDecoder(first_dec_with_lstm)

        scnd_dec_output_with_lstm = torch.cat((scnd_dec_output, lstm_outputs[2].view(-1, *lstm_outputs[2].shape[2:])), 1)
        

        out_3 = self.thirdDecoder(scnd_dec_output_with_lstm)
        output = out_3.reshape(batch_size, seq_len, *out_3.shape[1:])
        return self.sigmoid_layer(output)