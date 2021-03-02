from multiprocessing import Value
import torch
import torch.nn as nn

from model.provided_toolkit.pwcnet.pwcnet import PWCNet
from .rbpn import Net as RBPN
from .autoencoder_v4 import UNet
from .misc import Nearest
# from model.provided_toolkit.utils.metrics import AlignedL2_test as AlignedL2
from model.provided_toolkit.utils.metrics import AlignedL2
from model.engine.loss_functions import PITLoss


class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()
        self.flow_model = None
        
        self.burst_size = cfg.MODEL.BURST_SIZE
        self.preprocess = Nearest()

        self.use_flow = cfg.MODEL.USE_FLOW
        if self.use_flow:
            self.build_flow_model(cfg)
        self.flow_refine = cfg.MODEL.FLOW_REFINE
        if self.flow_refine:
            if not self.use_flow:
                raise ValueError("need to use flow inputs")
            self.FR_model = UNet(3 * 2 + 2, 2)
        self.model = RBPN(cfg)
        self.build_loss(cfg)
        
        self.denoise_burst = cfg.MODEL.DENOISE_BURST
        if self.denoise_burst:
            self.denoise_model = UNet(4, 4)

    def forward(self, data_dict):
        ret = self.pred(data_dict)
        loss = self.loss(ret)

        return loss
    
    def pred(self, data_dict):
        burst = data_dict['burst']
        burst = burst.cuda()
        
        if self.denoise_burst:
            size = burst.size()
            burst = burst + self.denoise_model(burst.view(-1, *size[2:])).view(size)
        
        if self.use_flow:
            aligned_burst = self.preprocess(burst)
            batch, burst_size, channel, height, width = aligned_burst.shape
            
            flow = self.flow_model(aligned_burst[:, 0, :, :, :].unsqueeze(1).expand(-1, burst_size, -1, -1, -1).reshape(-1, channel, height, width),
                                   aligned_burst.reshape(-1, channel, height, width))
            if self.flow_refine:
                flow += self.FR_model(torch.cat([flow,
                                                 aligned_burst[:, 0, :, :, :].unsqueeze(1).expand(-1, burst_size, -1, -1, -1).reshape(-1, channel, height, width),
                                                 aligned_burst.reshape(-1, channel, height, width)], axis=1))
            flow = flow.view(-1, burst_size, 2, height, width)
            flow[:, 0] = 0.0
            pred = self.model(burst, flow=flow)
        
        else:
            pred = self.model(burst)
            
        if torch.isnan(pred).sum().item() > 0 or torch.isinf(pred).sum().item() > 0:
            import pdb;pdb.set_trace()
        
        data_dict['pred'] = pred
        if self.denoise_burst:
            data_dict['denoised_burst'] = burst
        if self.flow_refine:
            data_dict['refined_flow'] = flow
            
        return data_dict
        
    def build_flow_model(self, cfg):
        self.flow_model = PWCNet(load_pretrained=True, weights_path=cfg.PWCNET_WEIGHTS)
        for param in self.flow_model.parameters():
            param.requires_grad = False
            
    def build_loss(self, cfg):
        self.loss_name = cfg.MODEL.LOSS
        self.l1 = nn.L1Loss()
        if cfg.MODEL.LOSS == 'l1':
            pass
        elif cfg.MODEL.LOSS == 'alignedl2':
            if not self.use_flow:
                self.build_flow_model(cfg)
            self.alignedl2 = AlignedL2(alignment_net=self.flow_model, sr_factor=4, boundary_ignore=10)
        elif cfg.MODEL.LOSS == 'pit':
            self.pit = PITLoss(cfg)
        else:
            raise ValueError(f"unknown loss function {cfg.MODEL.LOSS}")
        
    def loss(self, data_dict):
        loss = 0
        loss += self.recon_loss(data_dict['pred'].cuda(), data_dict['gt_frame'].cuda(), data_dict['burst'].cuda())
        if self.denoise_burst and 'gt_denoised_burst' in data_dict:
            loss += 0.1 / self.burst_size * self.l1(data_dict['denoised_burst'].cuda(), data_dict['gt_denoised_burst'].cuda()).cuda()
        if self.flow_refine and 'gt_flow' in data_dict:
            loss += 0.01 / self.burst_size * self.l1(data_dict['refined_flow'], data_dict['gt_flow'].cuda()).cuda()
        return loss
            
    def recon_loss(self, pred, target, x):
        if self.loss_name == 'l1':
            return self.l1(pred, target)
        elif self.loss_name == 'alignedl2':
            return self.alignedl2(pred, target, x)
        elif self.loss_name == 'pit':
            return self.pit(pred, target)
        else:
            raise ValueError
