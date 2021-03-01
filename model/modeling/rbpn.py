import torch
import torch.nn as nn
from torchvision.transforms import *

from .base_networks import *
from .dbpns import Net as DBPNS
from .misc import RGGB2channel, Nearest
from .extractors import *

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        #base_filter=256
        #feat=64
        
        # input_channel = cfg.MODEL.INPUT_CHANNEL
        if cfg.MODEL.PREPROCESS == "Nearest":
            input_channel = 3
            self.scale_factor = 4
            self.preprocess = Nearest()
            self.size_adjuster = nn.Upsample(scale_factor=1)
        elif cfg.MODEL.PREPROCESS == "RGGB2channel":
            input_channel = 4
            self.scale_factor = 8
            self.preprocess = RGGB2channel()
            self.size_adjuster = nn.Upsample(scale_factor=0.5)
        else:
            raise ValueError
        
        output_channel = cfg.MODEL.OUTPUT_CHANNEL
        # self.scale_factor = cfg.MODEL.SCALE_FACTOR
        base_filter = cfg.MODEL.BASE_FILTER
        feat = cfg.MODEL.FEAT
        n_resblock = cfg.MODEL.NUM_RESBLOCK
        burst_size = cfg.MODEL.BURST_SIZE

        self.use_flow = cfg.MODEL.USE_FLOW
        
        if self.scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif self.scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif self.scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        
        #Initial Feature Extraction
        if cfg.MODEL.EXTRACTOR_TYPE == 'original':
            self.extractor = OriginalExtractor(input_channel, base_filter, use_flow=self.use_flow)
        elif cfg.MODEL.EXTRACTOR_TYPE == 'normal':
            self.extractor = NormalExtractor(input_channel, base_filter, use_flow=self.use_flow)
        elif cfg.MODEL.EXTRACTOR_TYPE == 'deep' and cfg.MODEL.FIXUP_INIT:
            self.extractor = DeepExtractor_fixup_init(input_channel, base_filter, use_flow=self.use_flow)
        elif cfg.MODEL.EXTRACTOR_TYPE == 'deep' and not cfg.MODEL.FIXUP_INIT:
            self.extractor = DeepExtractor(input_channel, base_filter, use_flow=self.use_flow)
        elif cfg.MODEL.EXTRACTOR_TYPE == 'deform':
            self.extractor = DeformableExtractor(input_channel, base_filter, use_flow=self.use_flow)
        elif cfg.MODEL.EXTRACTOR_TYPE == 'deepdeform':
            self.extractor = DeepDeformableExtractor(input_channel, base_filter, use_flow=self.use_flow)
        elif cfg.MODEL.EXTRACTOR_TYPE == 'pcd_align':
            self.extractor = PCDAlignExtractor(input_channel, base_filter)
        else:
            raise NotImplementedError
        
        if cfg.MODEL.FIXUP_INIT:
            from .base_networks import ResnetBlock_fixup_init as ResnetBlock

        ###DBPNS
        self.DBPN = DBPNS(base_filter, feat, self.scale_factor)
                
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

            
    def forward(self, x, flow=None):
        x = self.preprocess(x)
        ### initial feature extraction
        features = self.extractor(x, flow)

        ####Projection
        Ht = []
        feat = features[0]
        for j in range(1, len(features)):
            h0 = self.DBPN(feat)
            h1 = self.res_feat1(features[j])
            
            e = h0-h1
            e = self.res_feat2(e)
            h = h0+e
            Ht.append(h)
            feat = self.res_feat3(h)
        
        ####Reconstruction
        out = torch.cat(Ht,1)
        output = self.output(out)

        return output
