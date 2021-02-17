import torch
import torch.nn as nn
from torchvision.transforms import *

from .base_networks import *
from .dbpns import Net as DBPNS
from .DCNv2.dcn_v2 import DCN_ID
from .misc import RGGB2channel, Nearest

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        #base_filter=256
        #feat=64
        
        # input_channel = cfg.MODEL.INPUT_CHANNEL
        if cfg.MODEL.PREPROCESS == "Nearest":
            input_channel = 3
            scale_factor = 4
            self.preprocess = Nearest()
        elif cfg.MODEL.PREPROCESS == "RGGB2channel":
            input_channel = 4
            scale_factor = 8
            self.preprocess = RGGB2channel()
        else:
            raise ValueError
        
        output_channel = cfg.MODEL.OUTPUT_CHANNEL
        # scale_factor = cfg.MODEL.SCALE_FACTOR
        base_filter = cfg.MODEL.BASE_FILTER
        feat = cfg.MODEL.FEAT
        n_resblock = cfg.MODEL.NUM_RESBLOCK
        burst_size = cfg.MODEL.BURST_SIZE

        self.use_flow = cfg.MODEL.USE_FLOW
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        self.init_conv = ConvBlock(input_channel, base_filter, 3, 1, 1, activation='prelu', norm=None)

        if self.use_flow:
            self.init_conv_for_offset = ConvBlock(input_channel*2 + 2, base_filter, 3, 1, 1, activation='prelu', norm=None)
        else:
            self.init_conv_for_offset = ConvBlock(input_channel*2, base_filter, 3, 1, 1, activation='prelu', norm=None)


        self.deform_conv = DCN_ID(base_filter, base_filter, base_filter, kernel_size=3, stride=1, padding=1)

        ###DBPNS
        self.DBPN = DBPNS(base_filter, feat, scale_factor)
                
        #Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)
        
        #Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)
        
        #Res-Block3
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)
        
        #Reconstruction
        self.output = ConvBlock((burst_size-1)*feat, output_channel, 3, 1, 1, activation=None, norm=None)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
                    
            
    def forward(self, x, flow=None):
        x = self.preprocess(x)
        
        base_frame = x[:, 0, :, :, :]
        neigbor_frame = x[:, 1:, :, :]

        ### initial feature extraction
        input_features = []
        for j in range(x.shape[1]):
            input_features.append(self.init_conv(x[:, j, :, :, :]))

        offset_features = []
        for j in range(neigbor_frame.shape[1]):
            if self.use_flow:
                offset_features.append(self.init_conv_for_offset(torch.cat((base_frame, neigbor_frame[:, j, :, :, :], flow[j]), 1)))
            else:
                offset_features.append(self.init_conv_for_offset(torch.cat((base_frame, neigbor_frame[:, j, :, :, :]), 1)))

        ### Projection
        Ht = []
        feat = input_features[0]
        for j in range(neigbor_frame.shape[1]):
            h0 = self.DBPN(feat)
            h1 = self.res_feat1(self.deform_conv(input_features[j+1], offset_features[j]))

            e = h0-h1
            e = self.res_feat2(e)
            h = h0+e
            Ht.append(h)
            feat = self.res_feat3(h)

        ### Reconstruction
        out = torch.cat(Ht, 1)
        output = self.output(out)

        return output