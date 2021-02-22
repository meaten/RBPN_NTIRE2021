import torch
import torch.nn as nn

from model.provided_toolkit.pwcnet.pwcnet import PWCNet
from .rbpn import Net as RBPN
# from .rbpn_deformconv import Net as Deformable_RBPN
from .misc import Nearest
from model.provided_toolkit.utils.metrics import AlignedL2


class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()

        self.preprocess = Nearest()

        self.use_flow = cfg.MODEL.USE_FLOW
        if self.use_flow:
            self.build_flow_model(cfg)
        
        if cfg.MODEL.TYPE == 'normal':
            self.model = RBPN(cfg)
        elif cfg.MODEL.TYPE == 'deform':
            self.model = Deformable_RBPN(cfg)

        if cfg.MODEL.LOSS == 'l1':
            self.loss_fn = nn.L1Loss()
        elif cfg.MODEL.TYPE == 'alignedl2':
            if not self.use_flow:
                self.build_flow_model(cfg)
            self.loss_fn = AlignedL2(alignment_net=self.flow_model, sr_factor=self.model.scale_factor, boundary_ignore=40)

    def forward(self, x, target):
        pred = self.pred(x)
        
        target = target.to('cuda')
        loss = self.loss_fn(pred, target)

        return loss
    
    def pred(self, x):
        x = x.to('cuda')
        if self.use_flow:
            alined_x = self.preprocess(x)
            batch, burst, channel, height, width = alined_x.shape
            flow = [torch.zeros(batch, 1, 2, height, width).to('cuda')]
            for j in range(1, alined_x.shape[1]):
                flow.append(self.flow_model(alined_x[:, 0, :, :, :], alined_x[:, j, :, :, :]))
            pred = self.model(x, flow=flow)
        
        else:
            pred = self.model(x)
            
        return pred
    
    def build_flow_model(self, cfg):
        self.flow_model = PWCNet(load_pretrained=True, weights_path=cfg.PWCNET_WEIGHTS)
        for param in self.flow_model.parameters():
            param.requires_grad = False
