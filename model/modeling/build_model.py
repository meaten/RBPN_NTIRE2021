import torch
import torch.nn as nn

from model.provided_toolkit.pwcnet.pwcnet import PWCNet
from .rbpn import Net as RBPN
from .misc import Nearest, RGGB2channel

class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()

        self.preprocess = Nearest()

        self.use_flow = cfg.MODEL.USE_FLOW
        if self.use_flow:
            self.flow_model = PWCNet(load_pretrained=True, weights_path=cfg.PWCNET_WEIGHTS)
            for param in self.flow_model.parameters():
                param.requires_grad = False

        self.model = RBPN(cfg)

        self.loss_fn = nn.L1Loss()
        

    def forward(self, x, target):
        x = x.to('cuda')
        target = target.to('cuda')
  
        if self.use_flow:
            alined_x = self.preprocess(x)
            batch, burst, channel, height, width = alined_x.shape
            flow = [torch.zeros(batch, 1, 2, height, width).to('cuda')]
            for j in range(1, alined_x.shape[1]):
                flow.append(self.flow_model(alined_x[:, 0, :, :, :], alined_x[:, j, :, :, :]))
            pred = self.model(x, flow=flow)   
        
        else:             
            pred = self.model(x)

        loss = self.loss_fn(pred, target)

        return loss