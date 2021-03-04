import torch
import torch.nn as nn

from .base_networks import *
from .DCNv2.dcn_v2 import DCN_ID

class OriginalExtractor(nn.Module):
    def __init__(self, input_channel, base_filter=64, use_flow=False):
        super(OriginalExtractor, self).__init__()

        self.use_flow = use_flow

        self.feat0 = ConvBlock(input_channel, base_filter, 3, 1, 1, activation='prelu', norm=None)
        
        if self.use_flow:
            self.feat1 = ConvBlock(input_channel*2 + 2, base_filter, 3, 1, 1, activation='prelu', norm=None)
        else:
            self.feat1 = ConvBlock(input_channel*2, base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.size_adjuster = nn.Upsample(scale_factor=0.5)
        
    def forward(self, x, flow=None):
        init_feature = self.feat0(x[:, 0, :, :, :])
        batch, burst_size, channel, height, width = x.shape
        if self.use_flow:
            features = self.feat1(torch.cat([x[:, 0, :, :, :].unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(-1, channel, height, width),
                                             x.reshape(-1, channel, height, width),
                                             self.size_adjuster(flow.view(-1, 2, height * 2, width * 2))], 1)).view(batch, burst_size, -1, height, width)
        else:
            features = self.feat1(torch.cat([x[:, 0, :, :, :].unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(-1, channel, height, width),
                                             x.reshape(-1, channel, height, width)], 1)).view(batch, burst_size, -1, height, width)
        features[:, 0] = init_feature

        return features.permute(1, 0, 2, 3, 4)


class NormalExtractor(nn.Module):
    def __init__(self, input_channel, base_filter, use_flow=False):
        super(NormalExtractor, self).__init__()

        self.use_flow = use_flow

        self.init_conv = nn.Sequential(
            ConvBlock(input_channel, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu'),
        )

        if self.use_flow:
            self.merge_conv = ConvBlock(base_filter*2 + 2, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu')
            self.size_adjuster = nn.Upsample(scale_factor=0.5)

        else:
            self.merge_conv = ConvBlock(base_filter*2, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu')


    def forward(self, x, flow=None):
        input_features = []
        for j in range(x.shape[1]):
            input_features.append(self.init_conv(x[:, j, :, :, :]))
        
        features = []
        for j in range(x.shape[1]):
            if self.use_flow:
                features.append(self.merge_conv(torch.cat((input_features[0], input_features[j], self.size_adjuster(flow[j])), 1)))
            else:
                features.append(self.merge_conv(torch.cat((input_features[0], input_features[j]), 1)))

        return features


class DeepExtractor(nn.Module):
    def __init__(self, input_channel, base_filter=64, use_flow=False):
        super(DeepExtractor, self).__init__()
        
        self.use_flow = use_flow

        self.init_conv = nn.Sequential(
            ConvBlock(input_channel, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu'),
            PyramidModule(base_filter, activation='prelu'),
        )

        if self.use_flow:
            self.merge_conv = ConvBlock(base_filter*2 + 2, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu')
            self.size_adjuster = nn.Upsample(scale_factor=0.5)

        else:
            self.merge_conv = ConvBlock(base_filter*2, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu')


    def forward(self, x, flow=None):
        input_features = []
        for j in range(x.shape[1]):
            input_features.append(self.init_conv(x[:, j, :, :, :]))
        
        features = []
        for j in range(x.shape[1]):
            if self.use_flow:
                features.append(self.merge_conv(torch.cat((input_features[0], input_features[j], self.size_adjuster(flow[j])), 1)))
            else:
                features.append(self.merge_conv(torch.cat((input_features[0], input_features[j]), 1)))

        return features
    
    
class DeepExtractor_fixup_init(nn.Module):
    def __init__(self, input_channel, base_filter=64, use_flow=False):
        super(DeepExtractor_fixup_init, self).__init__()
        
        self.use_flow = use_flow

        self.init_conv = nn.Sequential(
            ConvBlock(input_channel, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu'),
            PyramidModule_fixup_init(base_filter, activation='prelu'),
        )

        if self.use_flow:
            self.merge_conv = ConvBlock(base_filter*2 + 2, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu')
            self.size_adjuster = nn.Upsample(scale_factor=0.5)

        else:
            self.merge_conv = ConvBlock(base_filter*2, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu')

    def forward(self, x, flow=None):
        batch, burst_size, channel, height, width = x.shape
        input_feature = self.init_conv(x.reshape(-1, channel, height, width))

        input_feature_zero = input_feature.view(batch, burst_size, -1, height, width)[:, 0].unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(batch * burst_size, -1, height, width)
        
        if self.use_flow:
            features = self.merge_conv(torch.cat([input_feature_zero,
                                                                  input_feature,
                                                                  self.size_adjuster(flow.reshape(-1, 2, height * 2, width * 2))], 1))
        else:
            features = self.merge_conv(torch.cat((input_feature_zero, input_feature), 1))
            
        return features.view(batch, burst_size, *features.size()[1:]).permute(1, 0, 2, 3, 4)


class DeformableExtractor(nn.Module):
    def __init__(self, input_channel, base_filter=64, use_flow=False):
        super(DeformableExtractor, self).__init__()

        self.use_flow = use_flow

        self.init_conv = ConvBlock(input_channel, base_filter, 3, 1, 1, activation='prelu', norm=None)

        if self.use_flow:
            self.deform_conv = DCN_ID(base_filter, base_filter, base_filter*2 + 2, kernel_size=3, stride=1, padding=1)
            self.size_adjuster = nn.Upsample(scale_factor=0.5)
        else:
            self.deform_conv = DCN_ID(base_filter, base_filter, base_filter*2, kernel_size=3, stride=1, padding=1)


    def forward(self, x, flow=None):
        batch, burst_size, channel, height, width = x.shape
        input_feature = self.init_conv(x.reshape(-1, channel, height, width))

        input_feature_zero = input_feature.view(batch, burst_size, -1, height, width)[:, 0].unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(batch * burst_size, -1, height, width)
        
        if self.use_flow:
            features = self.deform_conv(input_feature, torch.cat([input_feature_zero,
                                                                  input_feature,
                                                                  self.size_adjuster(flow.reshape(-1, 2, height * 2, width * 2))], 1))
        else:
            features = self.deform_conv(input_feature, torch.cat((input_feature_zero, input_feature), 1))
            
        return features.view(batch, burst_size, *features.size()[1:]).permute(1, 0, 2, 3, 4)


class DeepDeformableExtractor(nn.Module):
    def __init__(self, input_channel, base_filter=64, use_flow=False):
        super(DeepDeformableExtractor, self).__init__()

        self.use_flow = use_flow

        self.init_conv = nn.Sequential(
            ConvBlock(input_channel, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu'),
            PyramidModule(base_filter, activation='prelu'),
        )

        if self.use_flow:
            self.deform_conv = DCN_ID(base_filter, base_filter, base_filter*2 + 2, kernel_size=3, stride=1, padding=1)
        else:
            self.deform_conv = DCN_ID(base_filter, base_filter, base_filter*2, kernel_size=3, stride=1, padding=1)

        self.size_adjuster = nn.Upsample(scale_factor=0.5)

    def forward(self, x, flow=None):
        batch, burst_size, channel, height, width = x.shape
        input_feature = self.init_conv(x.reshape(-1, channel, height, width))

        input_feature_zero = input_feature.view(batch, burst_size, -1, height, width)[:, 0].unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(batch * burst_size, -1, height, width)
        
        if self.use_flow:
            features = self.deform_conv(input_feature, torch.cat([input_feature_zero,
                                                                  input_feature,
                                                                  self.size_adjuster(flow.reshape(-1, 2, height * 2, width * 2))], 1))
        else:
            features = self.deform_conv(input_feature, torch.cat((input_feature_zero, input_feature), 1))
            
        return features.view(batch, burst_size, *features.size()[1:]).permute(1, 0, 2, 3, 4)


class PCDAlignExtractor(nn.Module):
    def __init__(self, input_channel, base_filter=64):
        super(PCDAlignExtractor, self).__init__()

        self.init_conv = nn.Sequential(
            ConvBlock(input_channel, base_filter, kernel_size=3, stride=1, padding=1, norm=None, activation='prelu'),
            PyramidModule(base_filter, activation='prelu'),
        )

        self.pcd_aligne_module = PCDAlignment(num_feat=base_filter, activation='prelu')


    def forward(self, x, flow=None):
        input_features = []
        for j in range(x.shape[1]):
            input_features.append(self.init_conv(x[:, j, :, :, :]))

        features = []
        for j in range(x.shape[1]):
            features.append(self.pcd_aligne_module(input_features[0], input_features[j]))

        return features
