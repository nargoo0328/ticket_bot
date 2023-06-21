import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, in_dim, in_channel):
        super(ResNet, self).__init__()
        self.inchannel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_channel, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.MaxPool2d(3,1,1)
        )
        layers = []
        for i in range(4):
            layers.append(self.make_layer(ResBlock, in_channel*(2**i), 2, stride=2))
        self.layers = nn.ModuleList(layers)    
  
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        features_map = []
        out = self.conv1(x)
        # C2
        for layer in self.layers:
            out = layer(out)
            features_map.append(out)
        return features_map
    
class FPN(nn.Module):
    def __init__(self,in_channel_list,out_channel):
        super(FPN, self).__init__()
        conv=[]
        out = []
        for in_channel in in_channel_list:
            conv.append(nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()))
            out.append(nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()))
        self.conv = nn.ModuleList(conv)
        self.out = nn.ModuleList(out)

    def forward(self,x):
        head_output=[]
        last_f = self.conv[-1](x[-1])
        head_output.append(self.out[-1](last_f))
        for i in range(len(x)-2,-1,-1):
            last_f = F.interpolate(last_f,scale_factor=(2,2),mode='nearest')
            last_f = self.conv[i](x[i]) + last_f
            head_output.append(self.out[i](last_f))
        return list(reversed(head_output))
