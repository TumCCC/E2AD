from models.resnet import resnet34, resnet50, wide_resnet50_2, resnext50_32x4d, resnet18, resnet50_dino
from models.resnet_decoder import resnet50_decoder, wide_resnet50_decoder, resnet34_decoder, resnext50_32x4d_decoder, resnet18_decoder, resnet50_decoder_sc, wide_resnet50_decoder_sc, resnet18_decoder_sc, resnet34_decoder_sc, resnet101_decoder_sc, resnext50_32x4d_decoder_sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision
import math
from copy import deepcopy

class SA(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(SA, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x

        return out

def zero_side(p, side=1):
    p[:, :, :side, :] = 0
    p[:, :, :, :side] = 0

    p[:, :, -side:, :] = 0
    p[:, :, :, -side:] = 0

    return p


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)



class Dict2Obj(dict):
    def __getattr__(self, key):
        if key not in self:
            return None
        else:
            value = self[key]
            if isinstance(value, dict):
                value = Dict2Obj(value)
            return value


class E2AD(nn.Module):
    def __init__(self,
                 img_size=256,
                 train_encoder=True,
                 stop_grad=True,
                 reshape=True,
                 bn_pretrain=False,
                 anomap_layer=[1, 2, 3],
                 ):
        super().__init__()

        self.edc_encoder   = resnet50(pretrained=True)
        self.edc_decoder_1 = resnet50_decoder_sc(pretrained=False, inplanes=[2048])
        self.edc_decoder_2 = resnet50_decoder_sc(pretrained=False, inplanes=[2048])

        self.edc_decoder_1.layer4 = None
        self.edc_decoder_2.layer4 = None
                
        self.train_encoder = train_encoder
        self.stop_grad = stop_grad
        self.reshape = reshape
        self.bn_pretrain = bn_pretrain
        self.anomap_layer = anomap_layer
        
        self.sa1 = SA(1024)
        self.sa2 = SA(512)
        
        self.sa3 = SA(1024)
        self.sa4 = SA(512)
        
        
        
        
    def forward(self, x):
        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()
            
        if self.bn_pretrain and self.edc_encoder.training:
            self.edc_encoder.eval()

        x_rot = torch.rot90(x, 2, (2, 3))
        B = x.shape[0]

        _, e1_1, e2_1, e3_1, e4_1 = self.edc_encoder(x)
        _, e1_2, e2_2, e3_2, e4_2 = self.edc_encoder(x_rot)        
        
        e4_1_global = self.edc_encoder.avgpool(e4_1).view(B,-1)
        e4_2_global = self.edc_encoder.avgpool(e4_2).view(B,-1)
        
        e2_1_sa = self.sa2(e2_1)
        e3_1_sa = self.sa1(e3_1)
        
        e2_2_sa = self.sa4(e2_2)
        e3_2_sa = self.sa3(e3_2)
        
      
        d1_1, d2_1, d3_1 = self.edc_decoder_1([e1_1, e2_1_sa,  e3_1_sa, e4_1])
        d1_2, d2_2, d3_2 = self.edc_decoder_2([e1_2, e2_2_sa,  e3_2_sa, e4_2])

        e1_1 = e1_1.detach()
        e2_1 = e2_1.detach()
        e3_1 = e3_1.detach()
        
        e1_2 = e1_2.detach()
        e2_2 = e2_2.detach()
        e3_2 = e3_2.detach()

        l1 = 1. - torch.cosine_similarity(d1_1.reshape(B, -1), e1_1.reshape(B, -1), dim=1).mean()
        l2 = 1. - torch.cosine_similarity(d2_1.reshape(B, -1), e2_1.reshape(B, -1), dim=1).mean()
        l3 = 1. - torch.cosine_similarity(d3_1.reshape(B, -1), e3_1.reshape(B, -1), dim=1).mean()
        
        l4 = 1. - torch.cosine_similarity(d1_2.reshape(B, -1), e1_2.reshape(B, -1), dim=1).mean()
        l5 = 1. - torch.cosine_similarity(d2_2.reshape(B, -1), e2_2.reshape(B, -1), dim=1).mean()
        l6 = 1. - torch.cosine_similarity(d3_2.reshape(B, -1), e3_2.reshape(B, -1), dim=1).mean()
        
        e_loss = 1. - torch.cosine_similarity(e4_1_global, e4_2_global, dim=1).mean()
            
        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1_1, e1_1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2_1, e2_1, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3_1, e3_1, dim=1).unsqueeze(1)
            
            p4 = 1. - torch.cosine_similarity(d1_2, e1_2, dim=1).unsqueeze(1)
            p5 = 1. - torch.cosine_similarity(d2_2, e2_2, dim=1).unsqueeze(1)
            p6 = 1. - torch.cosine_similarity(d3_2, e3_2, dim=1).unsqueeze(1)
        
        loss_1 = (l1 + l2 + l3)
        loss_2 = (l4 + l5 + l6)
        
        loss = (0.5 * loss_1 + 0.5 * loss_2)  + 0.5 * e_loss

        p2 = F.interpolate(p2, scale_factor=2)
        p3 = F.interpolate(p3, scale_factor=4)
        
        p5 = F.interpolate(p5, scale_factor=2)
        p6 = F.interpolate(p6, scale_factor=4)
          
        p_all_1 = [[p1, p2, p3][l - 1] for l in [1,2,3]]
        p_all_1 = torch.cat(p_all_1, dim=1).mean(dim=1, keepdim=True)
        
        p_all_2 = [[p4, p5, p6][l - 1] for l in [1,2,3]]
        p_all_2 = torch.cat(p_all_2, dim=1).mean(dim=1, keepdim=True)

        with torch.no_grad():
            e1_std = F.normalize(e1_1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(e2_1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(e3_1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {'loss': loss, 'p_all_1': p_all_1, 'p_all_2': p_all_2, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6,
                'e1_std': e1_std, 'e2_std': e2_std, 'e3_std': e3_std}
