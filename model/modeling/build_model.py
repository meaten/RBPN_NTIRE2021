import torch
import torch.nn as nn

from .rbpn import Net as RBPN

class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()

        self.model = RBPN(cfg)
        self.loss_fn = nn.L1Loss()

        self.rggb2rbg = RGGB2RBG()

    def forward(self, x, target):
        x = x.to('cuda')
        target = target.to('cuda')

        x = self.rggb2rbg(x)
        pred = self.model(x)
        loss = self.loss_fn(pred, target)

        return loss



class RGGB2RBG(object):
    def __init__(self):
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def __call__(self, x):
        batch, frame, channel, width, height = x.shape

        out = torch.empty((batch, frame, 3, width*2, height*2)).to(x.device)
        
        out[:, :, 0, :, :] = self.up(x[:, :, 0, :, :])
        out[:, :, 1, :, :] = ( self.up(x[:, :, 1, :, :]) + self.up(x[:, :, 2, :, :]) )/ 2
        out[:, :, 2, :, :] = self.up(x[:, :, 3, :, :])

        return out