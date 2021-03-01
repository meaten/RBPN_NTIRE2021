import torch
import torch.nn as nn

from model.provided_toolkit.pwcnet.pwcnet import PWCNet
from .rbpn import Net as RBPN
from .misc import Nearest
# from model.provided_toolkit.utils.metrics import AlignedL2_test as AlignedL2
from model.provided_toolkit.utils.metrics import AlignedL2
from model.engine.loss_functions import PITLoss


class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()
        self.flow_model = None
        
        self.preprocess = Nearest()

        self.use_flow = cfg.MODEL.USE_FLOW
        if self.use_flow:
            self.build_flow_model(cfg)
        self.model = RBPN(cfg)
        self.build_loss(cfg)

    def forward(self, x, target):
        x = x.to('cuda')
        target = target.to('cuda')
        pred = self.pred(x)
        loss = self.loss(pred, target, x)
        return loss
    
    def pred(self, x):
        if torch.isnan(x).sum().item() > 0 or torch.isinf(x).sum().item() > 0:
            import pdb;pdb.set_trace()
        
        if self.use_flow:
            aligned_x = self.preprocess(x)
            batch, burst, channel, height, width = aligned_x.shape
            flow = [torch.zeros(batch, 2, height, width).to('cuda')]
            for j in range(1, aligned_x.shape[1]):
                flow.append(self.flow_model(aligned_x[:, 0, :, :, :], aligned_x[:, j, :, :, :]))
            pred = self.model(x, flow=flow)
        
        else:
            pred = self.model(x)
            
        if torch.isnan(pred).sum().item() > 0 or torch.isinf(pred).sum().item() > 0:
            import pdb;pdb.set_trace()
        
        return pred
    
    def build_flow_model(self, cfg):
        self.flow_model = PWCNet(load_pretrained=True, weights_path=cfg.PWCNET_WEIGHTS)
        for param in self.flow_model.parameters():
            param.requires_grad = False
            
    def build_loss(self, cfg):
        self.loss_name = cfg.MODEL.LOSS
        if cfg.MODEL.LOSS == 'l1':
            self.l1 = nn.L1Loss()
        elif cfg.MODEL.LOSS == 'alignedl2':
            self.pretrain = True
            if not self.use_flow:
                self.build_flow_model(cfg)
            self.alignedl2 = AlignedL2(alignment_net=self.flow_model, sr_factor=4, boundary_ignore=10)
        elif cfg.MODEL.LOSS == 'pit':
            self.pit = PITLoss(cfg)
        else:
            raise ValueError(f"unknown loss function {cfg.MODEL.LOSS}")
            
    def loss(self, pred, target, x):
        if self.loss_name == 'l1':
            return self.l1(pred, target)
        elif self.loss_name == 'alignedl2':
            return self.alignedl2(pred, target, x)
        elif self.loss_name == 'pit':
            return self.pit(pred, target)
        else:
            raise ValueError
