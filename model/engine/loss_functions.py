import torch
import torch.nn as nn
from torchvision.models import vgg16


### Loss function proposed in iSeeBetter
class PITLoss(nn.Module):
    def __init__(self, cfg):
        super(PITLoss, self).__init__()

        self.mse_loss_fn = nn.MSELoss()
        self.mse_loss_weight = cfg.SOLVER.PITLOSS.MSE_WEIGHT

        self.vgg_loss_fn = VGGLoss()
        self.vgg_loss_weight = cfg.SOLVER.PITLOSS.VGG_WEIGHT

        self.tv_loss_fn = TVLoss()
        self.tv_loss_weight = cfg.SOLVER.PITLOSS.TV_WEIGHT

    def forward(self, pred, target):
        mse_loss = self.mse_loss_fn(pred, target) * self.mse_loss_weight
        vgg_loss = self.vgg_loss_fn(pred, target) * self.vgg_loss_weight
        tv_loss = self.tv_loss_fn(pred) * self.tv_loss_weight

        return mse_loss + vgg_loss + tv_loss



class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        
        self.vgg = nn.Sequential(*list(vgg16(pretrained=True).features[:31])).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False        

        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return self.mse(self.vgg(x), self.vgg(y))
