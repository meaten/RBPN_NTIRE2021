import torch
import torch.nn as nn

from .rbpn import Net as RBPN
from .rbpn_deformconv import Net as Deformable_RBPN

class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()

        if cfg.MODEL.TYPE == 'normal':
            self.model = RBPN(cfg)
        elif cfg.MODEL.TYPE == 'deform':
            self.model = Deformable_RBPN(cfg)
            
        self.loss_fn = nn.L1Loss()

    def forward(self, x, target):
        x = x.to('cuda')
        target = target.to('cuda')

        pred = self.model(x)
        loss = self.loss_fn(pred, target)

        return loss