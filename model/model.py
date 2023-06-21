import torch
import torch.nn as nn
from .backbone import ResNet, FPN
import math

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class captcha_model(nn.Module):
    def __init__(self, dim, nheads=1, num_encoder_layers=1, num_decoder_layers=1,cuda=True):
        super(captcha_model, self).__init__()
        self.dim = dim
        self.resnet = ResNet(1,dim)
        fpn_list = [dim * (2**i) for i in range(4)]
        self.fpn = FPN(fpn_list,dim)

        self.transformer = nn.Transformer(dim,nheads,num_encoder_layers,num_decoder_layers,128,batch_first=True,dropout=0.3)
        self.query = nn.Embedding(6,dim)
        self.pos_encode = positionalencoding2d(dim,6*2**0,8*2**0)
        if cuda:
            self.pos_encode = self.pos_encode.cuda()

        self.to_logits = nn.Linear(dim,27)

    def transformer_layer(self,x):
        b = x.shape[0]
        query = self.query.weight.clone()[None].expand(b,-1,-1)
        pos_encode = self.pos_encode.clone()[None].expand(b,-1,-1,-1)
        x = x + pos_encode
        x = x.reshape(b,-1,self.dim)
        query = query.reshape(b,-1,self.dim)
        x = self.transformer(x,query)
        return x
    
    def forward(self,x):
        x = self.resnet(x)
        x = self.fpn(x)
        x = self.transformer_layer(x[-1])
        x = self.to_logits(x)
        return x