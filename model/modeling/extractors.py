import torch
import torch.nn as nn

from .base_networks import *
from .DCNv2.dcn_v2 import DCN_ID

class NormalExtractor(nn.Module):
    def __init__(self, input_channel, base_filter=64, use_flow=False):
        super(NormalExtractor, self).__init__()

        self.use_flow = use_flow

        self.feat0 = ConvBlock(input_channel, base_filter, 3, 1, 1, activation='prelu', norm=None)
        
        if self.use_flow:
            self.feat1 = ConvBlock(input_channel*2 + 2, base_filter, 3, 1, 1, activation='prelu', norm=None)
        else:
            self.feat1 = ConvBlock(input_channel*2, base_filter, 3, 1, 1, activation='prelu', norm=None)
        
    def forward(self, x, flow=None):
        features = [self.feat0(x[:, 0, :, :, :])]
        for j in range(1, x.shape[1]):
            if self.use_flow:
                features.append(self.feat1(torch.cat((x[:, 0, :, :, :], x[:, j, :, :, :], self.size_adjuster[j+1]), 1)))
            else:
                features.append(self.feat1(torch.cat((x[:, 0, :, :, :], x[:, j, :, :, :]), 1)))

        return features


class DeformableExtractor(nn.Module):
    def __init__(self, input_channel, base_filter=64, use_flow=False):
        super(DeformableExtractor, self).__init__()

        self.use_flow = use_flow

        self.feat0 = ConvBlock(input_channel, base_filter, 3, 1, 1, activation='prelu', norm=None)

        if self.use_flow:
            self.init_conv_for_offset = ConvBlock(input_channel*2 + 2, base_filter, 3, 1, 1, activation='prelu', norm=None)
        else:
            self.init_conv_for_offset = ConvBlock(input_channel*2, base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.deform_conv = DCN_ID(base_filter, base_filter, kernel_size=3, stride=1, padding=1)


    def forward(self, x, flow):
        input_features = []
        for j in range(x.shape[1]):
            input_features.append(self.init_conv(x[:, j, :, :, :]))

        offset_features = []
        for j in range(x.shape[1]):
            if self.use_flow:
                offset_features.append(self.init_conv_for_offset(torch.cat((x[:, 0, :, :, :], x[:, j, :, :, :], self.size_adjuster(flow[j])), 1)))
            else:
                offset_features.append(self.init_conv_for_offset(torch.cat((x[:, 0, :, :, :], x[:, j, :, :, :]), 1)))

        features = []
        for j in range(x.shape[1]):
            features.append(self.deform_conv(input_features[j], offset_features[j]))

        return features



class PCDAligneExtractor(nn.Module):
    def __init__(self, input_channel, base_filter):
        super(PCDAligneExtractor, self).__init__()

        self.init_conv = nn.Sequential(
            ConvBlock(input_channel, base_filter, kernel_size=3, stride=1, padding=1, activation='prelu', norm=None),
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, activation='prelu', norm=None),
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, activation='prelu', norm=None),
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, activation='prelu', norm=None),
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, activation='prelu', norm=None),
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, activation='prelu', norm=None),
        )

        self.pdc_aligne_module = PDCAlignment()


    def forward(self, x, flow):
        input_features = []
        for j in range(x.shape[1]):
            input_features.append(self.init_conv(x[:, j, :, :, :]))

        features = []
        for j in range(x.shape[1]):
            features.append(self.pdc_aligne_module(input_features[0], input_features[j]))

        return features
